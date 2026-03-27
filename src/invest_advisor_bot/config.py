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
DEFAULT_AI_SIMULATED_PORTFOLIO_STATE_PATH = PROJECT_ROOT / "data" / "ai_simulated_portfolio.json"
DEFAULT_SECTOR_ROTATION_STATE_PATH = PROJECT_ROOT / "data" / "sector_rotation_state.json"
DEFAULT_REPORT_MEMORY_PATH = PROJECT_ROOT / "data" / "report_memory.json"
DEFAULT_RUNTIME_HISTORY_PATH = PROJECT_ROOT / "data" / "runtime_history.json"
DEFAULT_ANALYTICS_ROOT = PROJECT_ROOT / "data" / "analytics"
DEFAULT_GX_ROOT = PROJECT_ROOT / "data" / "great_expectations"
DEFAULT_EVIDENTLY_ROOT = PROJECT_ROOT / "data" / "evidently"
DEFAULT_BRAINTRUST_ROOT = PROJECT_ROOT / "data" / "braintrust"
DEFAULT_LANGFUSE_ROOT = PROJECT_ROOT / "data" / "langfuse"
DEFAULT_HUMAN_REVIEW_ROOT = PROJECT_ROOT / "data" / "human_reviews"
DEFAULT_QDRANT_ROOT = PROJECT_ROOT / "data" / "qdrant"
DEFAULT_FEATURE_STORE_ROOT = PROJECT_ROOT / "data" / "feature_store"
DEFAULT_BACKTESTING_ROOT = PROJECT_ROOT / "data" / "backtesting"
DEFAULT_EVENT_BUS_ROOT = PROJECT_ROOT / "data" / "event_bus"
DEFAULT_HOT_PATH_CACHE_ROOT = PROJECT_ROOT / "data" / "hot_path_cache"
DEFAULT_ANALYTICS_WAREHOUSE_ROOT = PROJECT_ROOT / "data" / "analytics_warehouse"
DEFAULT_SEMANTIC_ANALYST_ROOT = PROJECT_ROOT / "data" / "semantic_analyst"
DEFAULT_DBT_SEMANTIC_ROOT = PROJECT_ROOT / "data" / "dbt_semantic_layer"


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
    telegram_transport: str = Field(default="polling", validation_alias="TELEGRAM_TRANSPORT")
    telegram_webhook_url: str = Field(default="", validation_alias="TELEGRAM_WEBHOOK_URL")
    telegram_webhook_path: str = Field(default="/telegram/webhook", validation_alias="TELEGRAM_WEBHOOK_PATH")
    telegram_webhook_secret_token: str = Field(default="", validation_alias="TELEGRAM_WEBHOOK_SECRET_TOKEN")
    telegram_webhook_listen: str = Field(default="0.0.0.0", validation_alias="TELEGRAM_WEBHOOK_LISTEN")
    telegram_webhook_port: int = Field(
        default_factory=lambda: int(os.environ.get("PORT", "10000")),
        validation_alias="TELEGRAM_WEBHOOK_PORT",
    )
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
    cerebras_api_key: str = Field(default="", validation_alias="CEREBRAS_API_KEY")
    cerebras_models: str = Field(default="gpt-oss-120b,zai-glm-4.7", validation_alias="CEREBRAS_MODELS")
    cerebras_base_url: str = Field(default="https://api.cerebras.ai/v1", validation_alias="CEREBRAS_BASE_URL")
    cloudflare_account_id: str = Field(default="", validation_alias="CLOUDFLARE_ACCOUNT_ID")
    cloudflare_api_token: str = Field(default="", validation_alias="CLOUDFLARE_API_TOKEN")
    cloudflare_models: str = Field(
        default="@cf/meta/llama-3.1-8b-instruct-fast,@cf/meta/llama-3.1-8b-instruct",
        validation_alias="CLOUDFLARE_MODELS",
    )
    cloudflare_base_url: str = Field(default="", validation_alias="CLOUDFLARE_BASE_URL")
    huggingface_api_key: str = Field(default="", validation_alias="HUGGINGFACE_API_KEY")
    huggingface_models: str = Field(
        default="google/gemma-2-2b-it,Qwen/Qwen2.5-7B-Instruct-1M",
        validation_alias="HUGGINGFACE_MODELS",
    )
    huggingface_base_url: str = Field(
        default="https://router.huggingface.co/v1",
        validation_alias="HUGGINGFACE_BASE_URL",
    )
    tavily_api_key: str = Field(default="", validation_alias="TAVILY_API_KEY")
    exa_api_key: str = Field(default="", validation_alias="EXA_API_KEY")
    research_provider_order: str = Field(default="tavily,exa", validation_alias="RESEARCH_PROVIDER_ORDER")
    fmp_api_key: str = Field(default="", validation_alias="FMP_API_KEY")
    fmp_base_url: str = Field(default="https://financialmodelingprep.com/api/v3", validation_alias="FMP_BASE_URL")
    alpha_vantage_api_key: str = Field(default="", validation_alias="ALPHA_VANTAGE_API_KEY")
    finnhub_api_key: str = Field(default="", validation_alias="FINNHUB_API_KEY")
    fred_api_key: str = Field(default="", validation_alias="FRED_API_KEY")
    bls_api_key: str = Field(default="", validation_alias="BLS_API_KEY")
    eia_api_key: str = Field(default="", validation_alias="EIA_API_KEY")
    bea_api_key: str = Field(default="", validation_alias="BEA_API_KEY")
    ecb_api_base_url: str = Field(default="https://data-api.ecb.europa.eu/service/data", validation_alias="ECB_API_BASE_URL")
    ecb_series_map: str = Field(default="", validation_alias="ECB_SERIES_MAP")
    imf_api_base_url: str = Field(default="", validation_alias="IMF_API_BASE_URL")
    imf_series_map: str = Field(default="", validation_alias="IMF_SERIES_MAP")
    world_bank_api_base_url: str = Field(default="https://api.worldbank.org/v2", validation_alias="WORLD_BANK_API_BASE_URL")
    world_bank_countries: str = Field(default="EMU,EUU,JPN,CHN", validation_alias="WORLD_BANK_COUNTRIES")
    world_bank_indicator_map: str = Field(default="gdp_growth=NY.GDP.MKTP.KD.ZG,unemployment=SL.UEM.TOTL.ZS", validation_alias="WORLD_BANK_INDICATOR_MAP")
    global_macro_calendar_countries: str = Field(
        default="Euro Area,Japan,China,United Kingdom,Canada,Australia",
        validation_alias="GLOBAL_MACRO_CALENDAR_COUNTRIES",
    )
    global_macro_calendar_importance: int = Field(default=2, validation_alias="GLOBAL_MACRO_CALENDAR_IMPORTANCE")
    openbb_pat: str = Field(default="", validation_alias="OPENBB_PAT")
    openbb_base_url: str = Field(default="", validation_alias="OPENBB_BASE_URL")
    trading_economics_api_key: str = Field(default="", validation_alias="TRADING_ECONOMICS_API_KEY")
    polygon_api_key: str = Field(default="", validation_alias="POLYGON_API_KEY")
    polygon_base_url: str = Field(default="https://api.polygon.io", validation_alias="POLYGON_BASE_URL")
    polygon_options_chain_limit: int = Field(default=20, validation_alias="POLYGON_OPTIONS_CHAIN_LIMIT")
    cme_fedwatch_api_key: str = Field(default="", validation_alias="CME_FEDWATCH_API_KEY")
    cme_fedwatch_api_url: str = Field(default="", validation_alias="CME_FEDWATCH_API_URL")
    nasdaq_data_link_api_key: str = Field(default="", validation_alias="NASDAQ_DATA_LINK_API_KEY")
    nasdaq_data_link_base_url: str = Field(default="https://data.nasdaq.com/api/v3", validation_alias="NASDAQ_DATA_LINK_BASE_URL")
    nasdaq_data_link_datasets: str = Field(default="", validation_alias="NASDAQ_DATA_LINK_DATASETS")
    gdelt_context_base_url: str = Field(default="https://api.gdeltproject.org/api/v2/doc/doc", validation_alias="GDELT_CONTEXT_BASE_URL")
    gdelt_geo_base_url: str = Field(default="https://api.gdeltproject.org/api/v2/geo/geo", validation_alias="GDELT_GEO_BASE_URL")
    gdelt_query: str = Field(default="", validation_alias="GDELT_QUERY")
    gdelt_max_records: int = Field(default=25, validation_alias="GDELT_MAX_RECORDS")
    databento_api_key: str = Field(default="", validation_alias="DATABENTO_API_KEY")
    databento_equities_dataset: str = Field(default="", validation_alias="DATABENTO_EQUITIES_DATASET")
    databento_options_dataset: str = Field(default="", validation_alias="DATABENTO_OPTIONS_DATASET")
    databento_equities_schema: str = Field(default="mbp-10", validation_alias="DATABENTO_EQUITIES_SCHEMA")
    databento_options_schema: str = Field(default="mbp-10", validation_alias="DATABENTO_OPTIONS_SCHEMA")
    databento_lookback_minutes: int = Field(default=5, validation_alias="DATABENTO_LOOKBACK_MINUTES")
    live_stream_enabled: bool = Field(default=False, validation_alias="LIVE_STREAM_ENABLED")
    live_stream_dataset: str = Field(default="", validation_alias="LIVE_STREAM_DATASET")
    live_stream_schema: str = Field(default="trades", validation_alias="LIVE_STREAM_SCHEMA")
    live_stream_symbols: str = Field(default="SPY,QQQ,AAPL,MSFT,NVDA", validation_alias="LIVE_STREAM_SYMBOLS")
    live_stream_poll_interval_seconds: int = Field(default=60, validation_alias="LIVE_STREAM_POLL_INTERVAL_SECONDS")
    live_stream_max_events: int = Field(default=25, validation_alias="LIVE_STREAM_MAX_EVENTS")
    live_stream_sample_timeout_seconds: float = Field(default=3.0, validation_alias="LIVE_STREAM_SAMPLE_TIMEOUT_SECONDS")
    live_stream_spread_alert_bps: float = Field(default=25.0, validation_alias="LIVE_STREAM_SPREAD_ALERT_BPS")
    broker_sandbox_enabled: bool = Field(default=False, validation_alias="BROKER_SANDBOX_ENABLED")
    broker_provider: str = Field(default="alpaca", validation_alias="BROKER_PROVIDER")
    alpaca_api_key: str = Field(default="", validation_alias="ALPACA_API_KEY")
    alpaca_api_secret: str = Field(default="", validation_alias="ALPACA_API_SECRET")
    alpaca_base_url: str = Field(default="https://paper-api.alpaca.markets", validation_alias="ALPACA_BASE_URL")
    tradier_access_token: str = Field(default="", validation_alias="TRADIER_ACCESS_TOKEN")
    tradier_account_id: str = Field(default="", validation_alias="TRADIER_ACCOUNT_ID")
    tradier_base_url: str = Field(default="https://sandbox.tradier.com", validation_alias="TRADIER_BASE_URL")
    broker_timeout_seconds: float = Field(default=12.0, validation_alias="BROKER_TIMEOUT_SECONDS")
    cboe_trade_alert_enabled: bool = Field(default=False, validation_alias="CBOE_TRADE_ALERT_ENABLED")
    cboe_trade_alert_api_key: str = Field(default="", validation_alias="CBOE_TRADE_ALERT_API_KEY")
    cboe_trade_alert_base_url: str = Field(default="", validation_alias="CBOE_TRADE_ALERT_BASE_URL")
    order_flow_timeout_seconds: float = Field(default=12.0, validation_alias="ORDER_FLOW_TIMEOUT_SECONDS")
    sec_user_agent: str = Field(
        default="InvestAdvisorBot/0.2 support@example.com",
        validation_alias="SEC_USER_AGENT",
    )
    sec_13f_manager_ciks: str = Field(default="", validation_alias="SEC_13F_MANAGER_CIKS")
    policy_feed_enabled: bool = Field(default=True, validation_alias="POLICY_FEED_ENABLED")
    fed_speeches_feed_url: str = Field(
        default="https://www.federalreserve.gov/feeds/speeches.xml",
        validation_alias="FED_SPEECHES_FEED_URL",
    )
    fed_press_feed_url: str = Field(
        default="https://www.federalreserve.gov/feeds/press_all.xml",
        validation_alias="FED_PRESS_FEED_URL",
    )
    ecb_press_feed_url: str = Field(
        default="https://www.ecb.europa.eu/rss/press.html",
        validation_alias="ECB_PRESS_FEED_URL",
    )
    ecb_speeches_feed_url: str = Field(
        default="https://www.ecb.europa.eu/rss/speeches.html",
        validation_alias="ECB_SPEECHES_FEED_URL",
    )
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
    default_investor_profile: str = Field(default="growth", validation_alias="DEFAULT_INVESTOR_PROFILE")
    alert_state_path: Path = Field(default=DEFAULT_ALERT_STATE_PATH, validation_alias="ALERT_STATE_PATH")
    user_state_path: Path = Field(default=DEFAULT_USER_STATE_PATH, validation_alias="USER_STATE_PATH")
    portfolio_state_path: Path = Field(default=DEFAULT_PORTFOLIO_STATE_PATH, validation_alias="PORTFOLIO_STATE_PATH")
    ai_simulated_portfolio_enabled: bool = Field(default=True, validation_alias="AI_SIM_PORTFOLIO_ENABLED")
    ai_simulated_portfolio_state_path: Path = Field(
        default=DEFAULT_AI_SIMULATED_PORTFOLIO_STATE_PATH,
        validation_alias="AI_SIM_PORTFOLIO_STATE_PATH",
    )
    ai_simulated_portfolio_starting_cash_usd: float = Field(
        default=1000.0,
        validation_alias="AI_SIM_PORTFOLIO_STARTING_CASH_USD",
    )
    ai_simulated_portfolio_max_positions: int = Field(default=5, validation_alias="AI_SIM_PORTFOLIO_MAX_POSITIONS")
    ai_simulated_portfolio_max_position_pct: float = Field(
        default=0.25,
        validation_alias="AI_SIM_PORTFOLIO_MAX_POSITION_PCT",
    )
    ai_simulated_portfolio_min_cash_pct: float = Field(default=0.10, validation_alias="AI_SIM_PORTFOLIO_MIN_CASH_PCT")
    ai_simulated_portfolio_min_trade_notional_usd: float = Field(
        default=25.0,
        validation_alias="AI_SIM_PORTFOLIO_MIN_TRADE_NOTIONAL_USD",
    )
    ai_simulated_portfolio_rebalance_interval_minutes: int = Field(
        default=360,
        validation_alias="AI_SIM_PORTFOLIO_REBALANCE_INTERVAL_MINUTES",
    )
    ai_simulated_portfolio_core_tickers: str = Field(
        default="SPY,QQQ,VTI,VOO,GLD,IAU,TLT",
        validation_alias="AI_SIM_PORTFOLIO_CORE_TICKERS",
    )
    ai_simulated_portfolio_profile: str = Field(
        default="growth",
        validation_alias="AI_SIM_PORTFOLIO_PROFILE",
    )
    ai_simulated_portfolio_allowed_asset_types: str = Field(
        default="stock,etf,gold",
        validation_alias="AI_SIM_PORTFOLIO_ALLOWED_ASSET_TYPES",
    )
    ai_simulated_portfolio_allow_fractional: bool = Field(
        default=True,
        validation_alias="AI_SIM_PORTFOLIO_ALLOW_FRACTIONAL",
    )
    sector_rotation_state_path: Path = Field(
        default=DEFAULT_SECTOR_ROTATION_STATE_PATH,
        validation_alias="SECTOR_ROTATION_STATE_PATH",
    )
    report_memory_path: Path = Field(default=DEFAULT_REPORT_MEMORY_PATH, validation_alias="REPORT_MEMORY_PATH")
    runtime_history_path: Path = Field(default=DEFAULT_RUNTIME_HISTORY_PATH, validation_alias="RUNTIME_HISTORY_PATH")
    runtime_history_retention_days: int = Field(default=30, validation_alias="RUNTIME_HISTORY_RETENTION_DAYS")
    pgvector_enabled: bool = Field(default=True, validation_alias="PGVECTOR_ENABLED")
    qdrant_enabled: bool = Field(default=False, validation_alias="QDRANT_ENABLED")
    qdrant_url: str = Field(default="", validation_alias="QDRANT_URL")
    qdrant_api_key: str = Field(default="", validation_alias="QDRANT_API_KEY")
    qdrant_collection_name: str = Field(default="invest_advisor_thesis_memory", validation_alias="QDRANT_COLLECTION_NAME")
    qdrant_vector_size: int = Field(default=64, validation_alias="QDRANT_VECTOR_SIZE")
    qdrant_root_dir: Path = Field(default=DEFAULT_QDRANT_ROOT, validation_alias="QDRANT_ROOT_DIR")
    thesis_embedding_api_key: str = Field(default="", validation_alias="THESIS_EMBEDDING_API_KEY")
    thesis_embedding_base_url: str = Field(default="https://api.openai.com/v1", validation_alias="THESIS_EMBEDDING_BASE_URL")
    thesis_embedding_model: str = Field(default="text-embedding-3-small", validation_alias="THESIS_EMBEDDING_MODEL")
    thesis_embedding_timeout_seconds: float = Field(default=12.0, validation_alias="THESIS_EMBEDDING_TIMEOUT_SECONDS")
    thesis_rerank_enabled: bool = Field(default=True, validation_alias="THESIS_RERANK_ENABLED")
    thesis_memory_top_k: int = Field(default=3, validation_alias="THESIS_MEMORY_TOP_K")
    feature_store_enabled: bool = Field(default=False, validation_alias="FEATURE_STORE_ENABLED")
    feast_enabled: bool = Field(default=False, validation_alias="FEAST_ENABLED")
    feast_project_name: str = Field(default="invest_advisor_bot", validation_alias="FEAST_PROJECT_NAME")
    feature_store_root_dir: Path = Field(default=DEFAULT_FEATURE_STORE_ROOT, validation_alias="FEATURE_STORE_ROOT_DIR")
    backtesting_enabled: bool = Field(default=False, validation_alias="BACKTESTING_ENABLED")
    backtesting_root_dir: Path = Field(default=DEFAULT_BACKTESTING_ROOT, validation_alias="BACKTESTING_ROOT_DIR")
    backtesting_benchmark_ticker: str = Field(default="SPY", validation_alias="BACKTESTING_BENCHMARK_TICKER")
    backtesting_lookback_period: str = Field(default="6mo", validation_alias="BACKTESTING_LOOKBACK_PERIOD")
    backtesting_history_limit: int = Field(default=126, validation_alias="BACKTESTING_HISTORY_LIMIT")
    backtesting_min_history_points: int = Field(default=30, validation_alias="BACKTESTING_MIN_HISTORY_POINTS")
    event_bus_enabled: bool = Field(default=False, validation_alias="EVENT_BUS_ENABLED")
    event_bus_brokers: str = Field(default="", validation_alias="EVENT_BUS_BROKERS")
    event_bus_topic_prefix: str = Field(default="invest_advisor", validation_alias="EVENT_BUS_TOPIC_PREFIX")
    event_bus_root_dir: Path = Field(default=DEFAULT_EVENT_BUS_ROOT, validation_alias="EVENT_BUS_ROOT_DIR")
    event_bus_consumer_enabled: bool = Field(default=False, validation_alias="EVENT_BUS_CONSUMER_ENABLED")
    event_bus_consumer_group: str = Field(default="invest-advisor-bot", validation_alias="EVENT_BUS_CONSUMER_GROUP")
    event_bus_consumer_poll_interval_seconds: int = Field(default=60, validation_alias="EVENT_BUS_CONSUMER_POLL_INTERVAL_SECONDS")
    event_bus_consumer_batch_size: int = Field(default=100, validation_alias="EVENT_BUS_CONSUMER_BATCH_SIZE")
    hot_path_cache_enabled: bool = Field(default=False, validation_alias="HOT_PATH_CACHE_ENABLED")
    redis_url: str = Field(default="", validation_alias="REDIS_URL")
    hot_path_cache_stream_prefix: str = Field(default="invest_advisor", validation_alias="HOT_PATH_CACHE_STREAM_PREFIX")
    hot_path_cache_root_dir: Path = Field(default=DEFAULT_HOT_PATH_CACHE_ROOT, validation_alias="HOT_PATH_CACHE_ROOT_DIR")
    analytics_warehouse_enabled: bool = Field(default=False, validation_alias="ANALYTICS_WAREHOUSE_ENABLED")
    clickhouse_url: str = Field(default="", validation_alias="CLICKHOUSE_URL")
    clickhouse_database: str = Field(default="default", validation_alias="CLICKHOUSE_DATABASE")
    clickhouse_username: str = Field(default="", validation_alias="CLICKHOUSE_USERNAME")
    clickhouse_password: str = Field(default="", validation_alias="CLICKHOUSE_PASSWORD")
    analytics_warehouse_root_dir: Path = Field(default=DEFAULT_ANALYTICS_WAREHOUSE_ROOT, validation_alias="ANALYTICS_WAREHOUSE_ROOT_DIR")
    semantic_analyst_enabled: bool = Field(default=False, validation_alias="SEMANTIC_ANALYST_ENABLED")
    semantic_analyst_api_url: str = Field(default="", validation_alias="SEMANTIC_ANALYST_API_URL")
    semantic_analyst_api_key: str = Field(default="", validation_alias="SEMANTIC_ANALYST_API_KEY")
    semantic_analyst_model_name: str = Field(default="local-heuristic", validation_alias="SEMANTIC_ANALYST_MODEL_NAME")
    semantic_analyst_timeout_seconds: float = Field(default=12.0, validation_alias="SEMANTIC_ANALYST_TIMEOUT_SECONDS")
    semantic_analyst_root_dir: Path = Field(default=DEFAULT_SEMANTIC_ANALYST_ROOT, validation_alias="SEMANTIC_ANALYST_ROOT_DIR")
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
    macro_event_refresh_interval_minutes: int = Field(default=5, validation_alias="MACRO_EVENT_REFRESH_INTERVAL_MINUTES")
    macro_event_pre_window_minutes: int = Field(default=20, validation_alias="MACRO_EVENT_PRE_WINDOW_MINUTES")
    macro_event_post_window_minutes: int = Field(default=90, validation_alias="MACRO_EVENT_POST_WINDOW_MINUTES")
    macro_event_lookahead_hours: int = Field(default=12, validation_alias="MACRO_EVENT_LOOKAHEAD_HOURS")
    risk_vix_alert_threshold: float = Field(default=30.0, validation_alias="RISK_VIX_ALERT_THRESHOLD")
    earnings_alert_days_ahead: int = Field(default=14, validation_alias="EARNINGS_ALERT_DAYS_AHEAD")

    health_check_enabled: bool = Field(default=True, validation_alias="HEALTH_CHECK_ENABLED")
    health_check_host: str = Field(default="0.0.0.0", validation_alias="HEALTH_CHECK_HOST")
    health_check_port: int = Field(
        default_factory=lambda: int(os.environ.get("PORT", "10000")),
        validation_alias="HEALTH_CHECK_PORT",
    )
    health_alert_webhook_url: str = Field(default="", validation_alias="HEALTH_ALERT_WEBHOOK_URL")
    health_alert_webhook_secret: str = Field(default="", validation_alias="HEALTH_ALERT_WEBHOOK_SECRET")
    health_alert_interval_minutes: int = Field(default=5, validation_alias="HEALTH_ALERT_INTERVAL_MINUTES")
    health_alert_cooldown_minutes: int = Field(default=30, validation_alias="HEALTH_ALERT_COOLDOWN_MINUTES")
    health_alert_timeout_seconds: float = Field(default=8.0, validation_alias="HEALTH_ALERT_TIMEOUT_SECONDS")
    health_alert_retry_count: int = Field(default=3, validation_alias="HEALTH_ALERT_RETRY_COUNT")
    health_alert_retry_backoff_seconds: float = Field(default=1.5, validation_alias="HEALTH_ALERT_RETRY_BACKOFF_SECONDS")
    mlflow_tracking_uri: str = Field(default="", validation_alias="MLFLOW_TRACKING_URI")
    mlflow_experiment_name: str = Field(default="invest-advisor-bot", validation_alias="MLFLOW_EXPERIMENT_NAME")
    data_quality_enabled: bool = Field(default=True, validation_alias="DATA_QUALITY_ENABLED")
    data_quality_gx_enabled: bool = Field(default=False, validation_alias="DATA_QUALITY_GX_ENABLED")
    data_quality_min_market_assets: int = Field(default=3, validation_alias="DATA_QUALITY_MIN_MARKET_ASSETS")
    data_quality_min_macro_sources: int = Field(default=2, validation_alias="DATA_QUALITY_MIN_MACRO_SOURCES")
    data_quality_min_news_items: int = Field(default=1, validation_alias="DATA_QUALITY_MIN_NEWS_ITEMS")
    data_quality_min_research_items: int = Field(default=0, validation_alias="DATA_QUALITY_MIN_RESEARCH_ITEMS")
    data_quality_gx_root_dir: Path = Field(default=DEFAULT_GX_ROOT, validation_alias="DATA_QUALITY_GX_ROOT_DIR")
    analytics_store_enabled: bool = Field(default=True, validation_alias="ANALYTICS_STORE_ENABLED")
    analytics_store_root: Path = Field(default=DEFAULT_ANALYTICS_ROOT, validation_alias="ANALYTICS_STORE_ROOT")
    analytics_parquet_export_interval_seconds: int = Field(default=300, validation_alias="ANALYTICS_PARQUET_EXPORT_INTERVAL_SECONDS")
    analytics_runtime_snapshot_interval_seconds: int = Field(default=300, validation_alias="ANALYTICS_RUNTIME_SNAPSHOT_INTERVAL_SECONDS")
    evidently_enabled: bool = Field(default=False, validation_alias="EVIDENTLY_ENABLED")
    evidently_root_dir: Path = Field(default=DEFAULT_EVIDENTLY_ROOT, validation_alias="EVIDENTLY_ROOT_DIR")
    evidently_report_every_n_events: int = Field(default=10, validation_alias="EVIDENTLY_REPORT_EVERY_N_EVENTS")
    braintrust_enabled: bool = Field(default=False, validation_alias="BRAINTRUST_ENABLED")
    braintrust_api_key: str = Field(default="", validation_alias="BRAINTRUST_API_KEY")
    braintrust_api_url: str = Field(default="https://api.braintrust.dev", validation_alias="BRAINTRUST_API_URL")
    braintrust_project_name: str = Field(default="invest-advisor-bot", validation_alias="BRAINTRUST_PROJECT_NAME")
    braintrust_experiment_name: str = Field(default="production-evals", validation_alias="BRAINTRUST_EXPERIMENT_NAME")
    braintrust_root_dir: Path = Field(default=DEFAULT_BRAINTRUST_ROOT, validation_alias="BRAINTRUST_ROOT_DIR")
    braintrust_report_every_n_events: int = Field(default=10, validation_alias="BRAINTRUST_REPORT_EVERY_N_EVENTS")
    langfuse_enabled: bool = Field(default=False, validation_alias="LANGFUSE_ENABLED")
    langfuse_public_key: str = Field(default="", validation_alias="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: str = Field(default="", validation_alias="LANGFUSE_SECRET_KEY")
    langfuse_host: str = Field(default="https://cloud.langfuse.com", validation_alias="LANGFUSE_HOST")
    langfuse_root_dir: Path = Field(default=DEFAULT_LANGFUSE_ROOT, validation_alias="LANGFUSE_ROOT_DIR")
    human_review_enabled: bool = Field(default=False, validation_alias="HUMAN_REVIEW_ENABLED")
    human_review_root_dir: Path = Field(default=DEFAULT_HUMAN_REVIEW_ROOT, validation_alias="HUMAN_REVIEW_ROOT_DIR")
    human_review_low_confidence_threshold: float = Field(
        default=0.58,
        validation_alias="HUMAN_REVIEW_LOW_CONFIDENCE_THRESHOLD",
    )
    human_review_sample_every_n: int = Field(default=5, validation_alias="HUMAN_REVIEW_SAMPLE_EVERY_N")
    dbt_semantic_layer_enabled: bool = Field(default=False, validation_alias="DBT_SEMANTIC_LAYER_ENABLED")
    dbt_semantic_layer_root_dir: Path = Field(
        default=DEFAULT_DBT_SEMANTIC_ROOT,
        validation_alias="DBT_SEMANTIC_LAYER_ROOT_DIR",
    )
    dbt_semantic_project_name: str = Field(
        default="invest_advisor_bot_semantic",
        validation_alias="DBT_SEMANTIC_PROJECT_NAME",
    )
    dbt_semantic_target_schema: str = Field(
        default="analytics",
        validation_alias="DBT_SEMANTIC_TARGET_SCHEMA",
    )
    prefect_enabled: bool = Field(default=False, validation_alias="PREFECT_ENABLED")

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
        "broker_timeout_seconds",
        "order_flow_timeout_seconds",
        "live_stream_sample_timeout_seconds",
        "live_stream_spread_alert_bps",
        "thesis_embedding_timeout_seconds",
        "semantic_analyst_timeout_seconds",
        "health_alert_timeout_seconds",
        "health_alert_retry_backoff_seconds",
        "human_review_low_confidence_threshold",
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
        "macro_event_refresh_interval_minutes",
        "macro_event_pre_window_minutes",
        "macro_event_post_window_minutes",
        "macro_event_lookahead_hours",
        "earnings_alert_days_ahead",
        "maintenance_cleanup_interval_minutes",
        "stock_pick_evaluation_interval_minutes",
        "stock_pick_evaluation_horizon_days",
        "backup_interval_hours",
        "backup_retention_days",
        "thesis_memory_top_k",
        "burn_in_target_days",
        "sector_rotation_min_streak",
        "earnings_result_lookback_days",
        "health_check_port",
        "health_alert_interval_minutes",
        "health_alert_cooldown_minutes",
        "health_alert_retry_count",
        "polygon_options_chain_limit",
        "global_macro_calendar_importance",
        "gdelt_max_records",
        "data_quality_min_market_assets",
        "data_quality_min_macro_sources",
        "data_quality_min_news_items",
        "data_quality_min_research_items",
        "analytics_parquet_export_interval_seconds",
        "analytics_runtime_snapshot_interval_seconds",
        "evidently_report_every_n_events",
        "braintrust_report_every_n_events",
        "human_review_sample_every_n",
        "databento_lookback_minutes",
        "event_bus_consumer_poll_interval_seconds",
        "event_bus_consumer_batch_size",
        "live_stream_poll_interval_seconds",
        "live_stream_max_events",
        "qdrant_vector_size",
        "backtesting_history_limit",
        "backtesting_min_history_points",
    )
    @classmethod
    def validate_non_negative_int(cls, value: int) -> int:
        if value < 0:
            raise ValueError("Configured integer values must not be negative")
        return value

    @field_validator("default_investor_profile")
    @classmethod
    def validate_default_investor_profile(cls, value: str) -> str:
        normalized = normalize_profile_name(value, default="growth")
        if normalized not in {"conservative", "balanced", "growth"}:
            raise ValueError("DEFAULT_INVESTOR_PROFILE must be conservative, balanced, or growth")
        return normalized

    @field_validator("llm_provider")
    @classmethod
    def validate_llm_provider(cls, value: str) -> str:
        normalized = value.strip().casefold() or "auto"
        if normalized not in {
            "auto",
            "openai",
            "openrouter",
            "gemini",
            "groq",
            "github_models",
            "cerebras",
            "cloudflare",
            "huggingface",
        }:
            raise ValueError(
                "LLM_PROVIDER must be auto, openai, openrouter, gemini, groq, github_models, cerebras, cloudflare, or huggingface"
            )
        return normalized

    @field_validator("telegram_transport")
    @classmethod
    def validate_telegram_transport(cls, value: str) -> str:
        normalized = value.strip().casefold() or "polling"
        if normalized not in {"polling", "webhook"}:
            raise ValueError("TELEGRAM_TRANSPORT must be polling or webhook")
        return normalized

    @property
    def project_root(self) -> Path:
        return PROJECT_ROOT

    def validate_runtime(self) -> None:
        if not self.telegram_token.strip():
            raise ValueError("TELEGRAM_TOKEN is required")
        if not self.system_prompt_path.exists():
            raise ValueError(f"System prompt file not found: {self.system_prompt_path}")
        if self.telegram_transport == "webhook" and not self.telegram_webhook_url.strip():
            raise ValueError("TELEGRAM_WEBHOOK_URL is required when TELEGRAM_TRANSPORT=webhook")

    def llm_available(self) -> bool:
        return bool(
            self.llm_api_key.strip()
            or self.openrouter_api_key.strip()
            or self.gemini_api_key.strip()
            or self.groq_api_key.strip()
            or self.github_models_api_key.strip()
            or self.cerebras_api_key.strip()
            or self.cloudflare_api_token.strip()
            or self.huggingface_api_key.strip()
        )

    def research_available(self) -> bool:
        return bool(self.tavily_api_key.strip() or self.exa_api_key.strip())


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
