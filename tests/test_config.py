from __future__ import annotations

from pathlib import Path

import pytest

from invest_advisor_bot.config import Settings


def test_settings_accept_runtime_values(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("system", encoding="utf-8")

    monkeypatch.setenv("TELEGRAM_TOKEN", "token")
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    settings = Settings(_env_file=None, system_prompt_path=prompt_path, logs_dir=tmp_path / "logs")

    settings.validate_runtime()
    assert settings.llm_available() is False
    assert settings.research_available() is False
    assert settings.default_investor_profile == "growth"
    assert settings.project_root.name == "MT5AI"


def test_settings_detects_non_openai_llm_keys(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("system", encoding="utf-8")

    monkeypatch.setenv("TELEGRAM_TOKEN", "token")
    monkeypatch.setenv("GEMINI_API_KEY", "gem-key")
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-key")

    settings = Settings(_env_file=None, system_prompt_path=prompt_path, logs_dir=tmp_path / "logs")

    assert settings.llm_available() is True
    assert settings.research_available() is True


def test_settings_detects_extended_openai_compatible_llm_keys(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("system", encoding="utf-8")

    monkeypatch.setenv("TELEGRAM_TOKEN", "token")
    monkeypatch.setenv("CEREBRAS_API_KEY", "cerebras-key")
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "cloudflare-token")
    monkeypatch.setenv("HUGGINGFACE_API_KEY", "hf-key")

    settings = Settings(_env_file=None, system_prompt_path=prompt_path, logs_dir=tmp_path / "logs")

    assert settings.llm_available() is True


def test_settings_accepts_vector_feature_and_backtesting_configuration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("system", encoding="utf-8")

    monkeypatch.setenv("TELEGRAM_TOKEN", "token")
    monkeypatch.setenv("QDRANT_ENABLED", "true")
    monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
    monkeypatch.setenv("QDRANT_VECTOR_SIZE", "256")
    monkeypatch.setenv("THESIS_EMBEDDING_API_KEY", "embed-key")
    monkeypatch.setenv("THESIS_EMBEDDING_MODEL", "text-embedding-3-small")
    monkeypatch.setenv("THESIS_EMBEDDING_TIMEOUT_SECONDS", "9")
    monkeypatch.setenv("THESIS_RERANK_ENABLED", "true")
    monkeypatch.setenv("FEATURE_STORE_ENABLED", "true")
    monkeypatch.setenv("FEAST_ENABLED", "true")
    monkeypatch.setenv("BACKTESTING_ENABLED", "true")
    monkeypatch.setenv("BACKTESTING_HISTORY_LIMIT", "90")

    settings = Settings(_env_file=None, system_prompt_path=prompt_path, logs_dir=tmp_path / "logs")

    assert settings.qdrant_enabled is True
    assert settings.qdrant_url == "http://localhost:6333"
    assert settings.qdrant_vector_size == 256
    assert settings.thesis_embedding_api_key == "embed-key"
    assert settings.thesis_embedding_model == "text-embedding-3-small"
    assert settings.thesis_embedding_timeout_seconds == 9
    assert settings.thesis_rerank_enabled is True
    assert settings.feature_store_enabled is True
    assert settings.feast_enabled is True
    assert settings.backtesting_enabled is True
    assert settings.backtesting_history_limit == 90


def test_settings_accepts_streaming_warehouse_and_semantic_analyst_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("system", encoding="utf-8")

    monkeypatch.setenv("TELEGRAM_TOKEN", "token")
    monkeypatch.setenv("ECB_SERIES_MAP", "inflation_yoy=ICP/M.U2.N.000000.4.ANR")
    monkeypatch.setenv("IMF_API_BASE_URL", "https://example.test/imf/{series}")
    monkeypatch.setenv("IMF_SERIES_MAP", "global_growth_pct=NGDP_RPCH")
    monkeypatch.setenv("WORLD_BANK_INDICATOR_MAP", "gdp_growth=NY.GDP.MKTP.KD.ZG")
    monkeypatch.setenv("GLOBAL_MACRO_CALENDAR_COUNTRIES", "Japan,Euro Area,Canada")
    monkeypatch.setenv("GLOBAL_MACRO_CALENDAR_IMPORTANCE", "3")
    monkeypatch.setenv("EVENT_BUS_ENABLED", "true")
    monkeypatch.setenv("EVENT_BUS_BROKERS", "localhost:9092")
    monkeypatch.setenv("EVENT_BUS_CONSUMER_ENABLED", "true")
    monkeypatch.setenv("EVENT_BUS_CONSUMER_GROUP", "ops-group")
    monkeypatch.setenv("EVENT_BUS_CONSUMER_POLL_INTERVAL_SECONDS", "45")
    monkeypatch.setenv("EVENT_BUS_CONSUMER_BATCH_SIZE", "250")
    monkeypatch.setenv("HOT_PATH_CACHE_ENABLED", "true")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setenv("ANALYTICS_WAREHOUSE_ENABLED", "true")
    monkeypatch.setenv("CLICKHOUSE_URL", "http://localhost:8123")
    monkeypatch.setenv("SEMANTIC_ANALYST_ENABLED", "true")
    monkeypatch.setenv("SEMANTIC_ANALYST_MODEL_NAME", "ops-analyst")

    settings = Settings(_env_file=None, system_prompt_path=prompt_path, logs_dir=tmp_path / "logs")

    assert settings.ecb_series_map == "inflation_yoy=ICP/M.U2.N.000000.4.ANR"
    assert settings.imf_api_base_url == "https://example.test/imf/{series}"
    assert settings.world_bank_indicator_map == "gdp_growth=NY.GDP.MKTP.KD.ZG"
    assert settings.global_macro_calendar_countries == "Japan,Euro Area,Canada"
    assert settings.global_macro_calendar_importance == 3
    assert settings.event_bus_enabled is True
    assert settings.event_bus_brokers == "localhost:9092"
    assert settings.event_bus_consumer_enabled is True
    assert settings.event_bus_consumer_group == "ops-group"
    assert settings.event_bus_consumer_poll_interval_seconds == 45
    assert settings.event_bus_consumer_batch_size == 250
    assert settings.hot_path_cache_enabled is True
    assert settings.redis_url == "redis://localhost:6379/0"
    assert settings.analytics_warehouse_enabled is True
    assert settings.clickhouse_url == "http://localhost:8123"
    assert settings.semantic_analyst_enabled is True
    assert settings.semantic_analyst_model_name == "ops-analyst"


def test_settings_accepts_langfuse_human_review_and_policy_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("system", encoding="utf-8")

    monkeypatch.setenv("TELEGRAM_TOKEN", "token")
    monkeypatch.setenv("CBOE_TRADE_ALERT_ENABLED", "true")
    monkeypatch.setenv("CBOE_TRADE_ALERT_API_KEY", "order-key")
    monkeypatch.setenv("CBOE_TRADE_ALERT_BASE_URL", "https://flow.example.test")
    monkeypatch.setenv("ORDER_FLOW_TIMEOUT_SECONDS", "15")
    monkeypatch.setenv("SEC_13F_MANAGER_CIKS", "0001067983,0001166559")
    monkeypatch.setenv("POLICY_FEED_ENABLED", "true")
    monkeypatch.setenv("FED_SPEECHES_FEED_URL", "https://fed.example.test/speeches.xml")
    monkeypatch.setenv("LANGFUSE_ENABLED", "true")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "lf-public")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "lf-secret")
    monkeypatch.setenv("HUMAN_REVIEW_ENABLED", "true")
    monkeypatch.setenv("HUMAN_REVIEW_LOW_CONFIDENCE_THRESHOLD", "0.61")
    monkeypatch.setenv("HUMAN_REVIEW_SAMPLE_EVERY_N", "7")
    monkeypatch.setenv("DBT_SEMANTIC_LAYER_ENABLED", "true")
    monkeypatch.setenv("DBT_SEMANTIC_PROJECT_NAME", "advisor_semantic")
    monkeypatch.setenv("DBT_SEMANTIC_TARGET_SCHEMA", "analytics_prod")

    settings = Settings(_env_file=None, system_prompt_path=prompt_path, logs_dir=tmp_path / "logs")

    assert settings.cboe_trade_alert_enabled is True
    assert settings.cboe_trade_alert_base_url == "https://flow.example.test"
    assert settings.order_flow_timeout_seconds == 15
    assert settings.sec_13f_manager_ciks == "0001067983,0001166559"
    assert settings.policy_feed_enabled is True
    assert settings.fed_speeches_feed_url == "https://fed.example.test/speeches.xml"
    assert settings.langfuse_enabled is True
    assert settings.langfuse_public_key == "lf-public"
    assert settings.human_review_enabled is True
    assert settings.human_review_low_confidence_threshold == 0.61
    assert settings.human_review_sample_every_n == 7
    assert settings.dbt_semantic_layer_enabled is True
    assert settings.dbt_semantic_project_name == "advisor_semantic"
    assert settings.dbt_semantic_target_schema == "analytics_prod"
