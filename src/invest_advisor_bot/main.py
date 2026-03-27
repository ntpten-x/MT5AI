from __future__ import annotations

import sys
import asyncio
import importlib
from contextlib import suppress
from typing import Any

from loguru import logger

from invest_advisor_bot.analytics_store import AnalyticsStore
from invest_advisor_bot.analytics_warehouse import AnalyticsWarehouse
from invest_advisor_bot.backtesting import BacktestingEngine
from invest_advisor_bot.braintrust_observer import BraintrustObserver
from invest_advisor_bot.bot.alert_state import AlertStateStore
from invest_advisor_bot.bot.ai_simulated_portfolio_state import AISimulatedPortfolioStateStore
from invest_advisor_bot.bot.backup_manager import BackupManager
from invest_advisor_bot.bot.health_check import (
    set_health_details_provider,
    start_health_check_server,
    stop_health_check_server,
)
from invest_advisor_bot.bot.portfolio_state import PortfolioStateStore
from invest_advisor_bot.bot.report_memory_state import ReportMemoryStore
from invest_advisor_bot.bot.runtime_history_store import RuntimeHistoryStore
from invest_advisor_bot.bot.sector_rotation_state import SectorRotationStateStore
from invest_advisor_bot.bot.telegram_app import create_application
from invest_advisor_bot.bot.user_state import UserStateStore
from invest_advisor_bot.config import Settings, get_settings
from invest_advisor_bot.data_quality import ReasoningDataQualityGate
from invest_advisor_bot.evidently_observer import EvidentlyObserver
from invest_advisor_bot.event_bus import EventBus
from invest_advisor_bot.event_bus_worker import EventBusConsumerWorker
from invest_advisor_bot.feature_store import FeatureStoreBridge
from invest_advisor_bot.hot_path_cache import HotPathCache
from invest_advisor_bot.human_review_store import HumanReviewStore
from invest_advisor_bot.langfuse_observer import LangfuseObserver
from invest_advisor_bot.mlflow_observer import MLflowObserver
from invest_advisor_bot.observability import log_event
from invest_advisor_bot.orchestration.prefect_flows import WorkflowOrchestrator
from invest_advisor_bot.providers.broker_client import ExecutionSandboxClient
from invest_advisor_bot.providers.llm_client import build_default_llm_client
from invest_advisor_bot.providers.market_data_client import MarketDataClient
from invest_advisor_bot.providers.live_market_stream import LiveMarketStreamClient
from invest_advisor_bot.providers.microstructure_client import MicrostructureClient
from invest_advisor_bot.providers.news_client import NewsClient
from invest_advisor_bot.providers.order_flow_client import OrderFlowClient
from invest_advisor_bot.providers.ownership_client import OwnershipIntelligenceClient
from invest_advisor_bot.providers.policy_feed_client import PolicyFeedClient
from invest_advisor_bot.providers.research_client import ResearchClient
from invest_advisor_bot.providers.transcript_client import EarningsTranscriptClient
from invest_advisor_bot.runtime_diagnostics import diagnostics
from invest_advisor_bot.runtime_status import collect_runtime_snapshot, sync_service_diagnostics
from invest_advisor_bot.semantic_analyst import SemanticAnalyst
from invest_advisor_bot.services.recommendation_service import RecommendationService
from invest_advisor_bot.services.ai_simulated_portfolio import AISimulatedPortfolioService
from invest_advisor_bot.thesis_vector_store import ThesisVectorStore
from invest_advisor_bot.dbt_semantic_layer import DbtSemanticLayer


def configure_logging(settings: Settings) -> None:
    settings.logs_dir.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(
        sys.stderr,
        level=settings.log_level,
        backtrace=False,
        diagnose=False,
    )
    logger.add(
        settings.logs_dir / "invest_advisor_bot.jsonl",
        level=settings.log_level,
        rotation=settings.log_rotation_size,
        retention=settings.log_retention,
        backtrace=False,
        diagnose=False,
        encoding="utf-8",
        serialize=True,
    )
    logger.add(
        settings.logs_dir / "invest_advisor_bot.log",
        level=settings.log_level,
        rotation=settings.log_rotation_size,
        retention=settings.log_retention,
        backtrace=False,
        diagnose=False,
        encoding="utf-8",
    )


def build_application(settings: Settings, *, database_url: str | None = None):
    effective_database_url = settings.database_url if database_url is None else database_url
    alert_state_store = AlertStateStore(
        path=settings.alert_state_path,
        suppression_minutes=settings.alert_suppression_minutes,
        database_url=effective_database_url,
    )
    user_state_store = UserStateStore(path=settings.user_state_path, database_url=effective_database_url)
    portfolio_state_store = PortfolioStateStore(
        path=settings.portfolio_state_path,
        database_url=effective_database_url,
    )
    ai_simulated_portfolio_state_store = AISimulatedPortfolioStateStore(
        path=settings.ai_simulated_portfolio_state_path,
        database_url=effective_database_url,
    )
    sector_rotation_state_store = SectorRotationStateStore(
        path=settings.sector_rotation_state_path,
        database_url=effective_database_url,
    )
    report_memory_store = ReportMemoryStore(path=settings.report_memory_path, database_url=effective_database_url)
    runtime_history_store = RuntimeHistoryStore(
        path=settings.runtime_history_path,
        database_url=effective_database_url,
        retention_days=settings.runtime_history_retention_days,
        pgvector_enabled=settings.pgvector_enabled,
    )
    backup_manager = BackupManager(
        backup_dir=settings.backup_dir,
        database_url=effective_database_url,
        retention_days=settings.backup_retention_days,
    )
    diagnostics.attach_history_store(runtime_history_store)
    analytics_store = AnalyticsStore(
        root_dir=settings.analytics_store_root,
        enabled=settings.analytics_store_enabled,
        parquet_export_interval_seconds=settings.analytics_parquet_export_interval_seconds,
        runtime_snapshot_interval_seconds=settings.analytics_runtime_snapshot_interval_seconds,
    )
    analytics_warehouse = AnalyticsWarehouse(
        root_dir=settings.analytics_warehouse_root_dir,
        enabled=settings.analytics_warehouse_enabled,
        clickhouse_url=settings.clickhouse_url,
        database=settings.clickhouse_database,
        username=settings.clickhouse_username,
        password=settings.clickhouse_password,
    )
    event_bus = EventBus(
        root_dir=settings.event_bus_root_dir,
        enabled=settings.event_bus_enabled,
        brokers=settings.event_bus_brokers,
        topic_prefix=settings.event_bus_topic_prefix,
    )
    hot_path_cache = HotPathCache(
        root_dir=settings.hot_path_cache_root_dir,
        enabled=settings.hot_path_cache_enabled,
        redis_url=settings.redis_url,
        stream_prefix=settings.hot_path_cache_stream_prefix,
    )
    event_bus_consumer = EventBusConsumerWorker(
        root_dir=settings.event_bus_root_dir,
        enabled=settings.event_bus_consumer_enabled,
        brokers=settings.event_bus_brokers,
        topic_prefix=settings.event_bus_topic_prefix,
        consumer_group=settings.event_bus_consumer_group,
        batch_size=settings.event_bus_consumer_batch_size,
        poll_timeout_seconds=settings.event_bus_consumer_poll_interval_seconds,
        analytics_warehouse=analytics_warehouse,
        hot_path_cache=hot_path_cache,
    )
    thesis_vector_store = ThesisVectorStore(
        root_dir=settings.qdrant_root_dir,
        enabled=settings.qdrant_enabled,
        qdrant_url=settings.qdrant_url,
        qdrant_api_key=settings.qdrant_api_key,
        collection_name=settings.qdrant_collection_name,
        vector_size=settings.qdrant_vector_size,
        embedding_api_key=settings.thesis_embedding_api_key,
        embedding_base_url=settings.thesis_embedding_base_url,
        embedding_model=settings.thesis_embedding_model,
        embedding_timeout_seconds=settings.thesis_embedding_timeout_seconds,
        rerank_enabled=settings.thesis_rerank_enabled,
    )
    feature_store = FeatureStoreBridge(
        root_dir=settings.feature_store_root_dir,
        enabled=settings.feature_store_enabled,
        feast_enabled=settings.feast_enabled,
        project_name=settings.feast_project_name,
    )
    backtesting_engine = BacktestingEngine(
        root_dir=settings.backtesting_root_dir,
        enabled=settings.backtesting_enabled,
        benchmark_ticker=settings.backtesting_benchmark_ticker,
        lookback_period=settings.backtesting_lookback_period,
        history_limit=settings.backtesting_history_limit,
        min_history_points=settings.backtesting_min_history_points,
    )
    semantic_analyst = SemanticAnalyst(
        root_dir=settings.semantic_analyst_root_dir,
        warehouse=analytics_warehouse,
        enabled=settings.semantic_analyst_enabled,
        api_url=settings.semantic_analyst_api_url,
        api_key=settings.semantic_analyst_api_key,
        model_name=settings.semantic_analyst_model_name,
        timeout_seconds=settings.semantic_analyst_timeout_seconds,
    )
    dbt_semantic_layer = DbtSemanticLayer(
        root_dir=settings.dbt_semantic_layer_root_dir,
        enabled=settings.dbt_semantic_layer_enabled,
        project_name=settings.dbt_semantic_project_name,
        target_schema=settings.dbt_semantic_target_schema,
    )
    dbt_semantic_layer.sync(warehouse=analytics_warehouse)
    langfuse_observer = LangfuseObserver(
        root_dir=settings.langfuse_root_dir,
        enabled=settings.langfuse_enabled,
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_host,
    )
    human_review_store = HumanReviewStore(
        root_dir=settings.human_review_root_dir,
        enabled=settings.human_review_enabled,
        low_confidence_threshold=settings.human_review_low_confidence_threshold,
        sample_every_n=settings.human_review_sample_every_n,
    )
    broker_client = ExecutionSandboxClient(
        provider=settings.broker_provider,
        enabled=settings.broker_sandbox_enabled,
        api_key=settings.alpaca_api_key,
        api_secret=settings.alpaca_api_secret,
        base_url=settings.alpaca_base_url,
        tradier_access_token=settings.tradier_access_token,
        tradier_account_id=settings.tradier_account_id,
        tradier_base_url=settings.tradier_base_url,
        timeout_seconds=settings.broker_timeout_seconds,
    )
    market_data_client = MarketDataClient(
        cache_ttl_seconds=settings.market_cache_ttl_seconds,
        alpha_vantage_api_key=settings.alpha_vantage_api_key,
        finnhub_api_key=settings.finnhub_api_key,
        fred_api_key=settings.fred_api_key,
        bls_api_key=settings.bls_api_key,
        eia_api_key=settings.eia_api_key,
        bea_api_key=settings.bea_api_key,
        ecb_api_base_url=settings.ecb_api_base_url,
        ecb_series_map=settings.ecb_series_map,
        imf_api_base_url=settings.imf_api_base_url,
        imf_series_map=settings.imf_series_map,
        world_bank_api_base_url=settings.world_bank_api_base_url,
        world_bank_countries=[item.strip() for item in settings.world_bank_countries.split(",") if item.strip()],
        world_bank_indicator_map=settings.world_bank_indicator_map,
        global_macro_calendar_countries=[item.strip() for item in settings.global_macro_calendar_countries.split(",") if item.strip()],
        global_macro_calendar_importance=settings.global_macro_calendar_importance,
        openbb_pat=settings.openbb_pat,
        openbb_base_url=settings.openbb_base_url,
        polygon_api_key=settings.polygon_api_key,
        polygon_base_url=settings.polygon_base_url,
        polygon_options_chain_limit=settings.polygon_options_chain_limit,
        cme_fedwatch_api_key=settings.cme_fedwatch_api_key,
        cme_fedwatch_api_url=settings.cme_fedwatch_api_url,
        nasdaq_data_link_api_key=settings.nasdaq_data_link_api_key,
        nasdaq_data_link_base_url=settings.nasdaq_data_link_base_url,
        nasdaq_data_link_datasets=[item.strip() for item in settings.nasdaq_data_link_datasets.split(",") if item.strip()],
        gdelt_context_base_url=settings.gdelt_context_base_url,
        gdelt_geo_base_url=settings.gdelt_geo_base_url,
        gdelt_query=settings.gdelt_query,
        gdelt_max_records=settings.gdelt_max_records,
        tradier_access_token=settings.tradier_access_token,
        tradier_base_url=settings.tradier_base_url,
        trading_economics_api_key=settings.trading_economics_api_key,
        sec_user_agent=settings.sec_user_agent,
        provider_order=[item.strip() for item in settings.market_data_provider_order.split(",") if item.strip()],
        http_timeout_seconds=settings.market_data_http_timeout_seconds,
    )
    news_client = NewsClient(
        timeout=settings.news_timeout_seconds,
        cache_ttl_seconds=settings.news_cache_ttl_seconds,
    )
    transcript_client = EarningsTranscriptClient(
        api_key=settings.fmp_api_key,
        base_url=settings.fmp_base_url,
        alpha_vantage_api_key=settings.alpha_vantage_api_key,
        timeout_seconds=settings.news_timeout_seconds,
        cache_ttl_seconds=max(settings.news_cache_ttl_seconds, 3600),
    )
    microstructure_client = MicrostructureClient(
        api_key=settings.databento_api_key,
        equities_dataset=settings.databento_equities_dataset,
        options_dataset=settings.databento_options_dataset,
        equities_schema=settings.databento_equities_schema,
        options_schema=settings.databento_options_schema,
        lookback_minutes=settings.databento_lookback_minutes,
    )
    order_flow_client = OrderFlowClient(
        enabled=settings.cboe_trade_alert_enabled,
        api_key=settings.cboe_trade_alert_api_key,
        base_url=settings.cboe_trade_alert_base_url,
        timeout_seconds=settings.order_flow_timeout_seconds,
        cache_ttl_seconds=max(settings.market_cache_ttl_seconds, 300),
    )
    ownership_client = OwnershipIntelligenceClient(
        sec_user_agent=settings.sec_user_agent,
        manager_ciks=[item.strip() for item in settings.sec_13f_manager_ciks.split(",") if item.strip()],
        timeout_seconds=settings.market_data_http_timeout_seconds,
        cache_ttl_seconds=max(settings.market_cache_ttl_seconds, 3600),
    )
    policy_feed_client = PolicyFeedClient(
        enabled=settings.policy_feed_enabled,
        fed_speeches_feed_url=settings.fed_speeches_feed_url,
        fed_press_feed_url=settings.fed_press_feed_url,
        ecb_press_feed_url=settings.ecb_press_feed_url,
        ecb_speeches_feed_url=settings.ecb_speeches_feed_url,
        timeout_seconds=settings.news_timeout_seconds,
        cache_ttl_seconds=max(settings.news_cache_ttl_seconds, 900),
    )
    live_market_stream_client = LiveMarketStreamClient(
        enabled=settings.live_stream_enabled,
        api_key=settings.databento_api_key,
        dataset=settings.live_stream_dataset,
        schema=settings.live_stream_schema,
        max_events_per_poll=settings.live_stream_max_events,
        sample_timeout_seconds=settings.live_stream_sample_timeout_seconds,
    )
    workflow_orchestrator = WorkflowOrchestrator(enabled=settings.prefect_enabled)
    llm_client = build_default_llm_client(
        llm_api_key=settings.llm_api_key,
        llm_model=settings.llm_model,
        llm_base_url=settings.llm_base_url,
        llm_timeout_seconds=settings.llm_timeout_seconds,
        llm_max_output_tokens=settings.llm_max_output_tokens,
        llm_organization=settings.llm_organization,
        llm_project=settings.llm_project,
        llm_provider=settings.llm_provider,
        llm_provider_order=[item.strip() for item in settings.llm_provider_order.split(",") if item.strip()],
        llm_model_fallbacks=[item.strip() for item in settings.llm_model_fallbacks.split(",") if item.strip()],
        openrouter_api_key=settings.openrouter_api_key,
        openrouter_models=[item.strip() for item in settings.openrouter_models.split(",") if item.strip()],
        openrouter_base_url=settings.openrouter_base_url,
        openrouter_http_referer=settings.openrouter_http_referer,
        openrouter_app_title=settings.openrouter_app_title,
        gemini_api_key=settings.gemini_api_key,
        gemini_models=[item.strip() for item in settings.gemini_models.split(",") if item.strip()],
        gemini_base_url=settings.gemini_base_url,
        groq_api_key=settings.groq_api_key,
        groq_models=[item.strip() for item in settings.groq_models.split(",") if item.strip()],
        groq_base_url=settings.groq_base_url,
        github_models_api_key=settings.github_models_api_key,
        github_models=[item.strip() for item in settings.github_models.split(",") if item.strip()],
        github_models_base_url=settings.github_models_base_url,
        github_models_api_version=settings.github_models_api_version,
        cerebras_api_key=settings.cerebras_api_key,
        cerebras_models=[item.strip() for item in settings.cerebras_models.split(",") if item.strip()],
        cerebras_base_url=settings.cerebras_base_url,
        cloudflare_account_id=settings.cloudflare_account_id,
        cloudflare_api_token=settings.cloudflare_api_token,
        cloudflare_models=[item.strip() for item in settings.cloudflare_models.split(",") if item.strip()],
        cloudflare_base_url=settings.cloudflare_base_url,
        huggingface_api_key=settings.huggingface_api_key,
        huggingface_models=[item.strip() for item in settings.huggingface_models.split(",") if item.strip()],
        huggingface_base_url=settings.huggingface_base_url,
    )
    research_client = ResearchClient(
        tavily_api_key=settings.tavily_api_key,
        exa_api_key=settings.exa_api_key,
        provider_order=[item.strip() for item in settings.research_provider_order.split(",") if item.strip()],
        timeout=settings.news_timeout_seconds,
        cache_ttl_seconds=settings.news_cache_ttl_seconds,
    )
    recommendation_service = RecommendationService(
        llm_client=llm_client,
        system_prompt_path=settings.system_prompt_path,
        default_investor_profile=settings.default_investor_profile,  # type: ignore[arg-type]
        runtime_history_store=runtime_history_store,
        thesis_memory_top_k=settings.thesis_memory_top_k,
        data_quality_gate=ReasoningDataQualityGate(
            enabled=settings.data_quality_enabled,
            gx_enabled=settings.data_quality_gx_enabled,
            min_market_assets=settings.data_quality_min_market_assets,
            min_macro_sources=settings.data_quality_min_macro_sources,
            min_news_items=settings.data_quality_min_news_items,
            min_research_items=settings.data_quality_min_research_items,
            gx_root_dir=settings.data_quality_gx_root_dir,
        ),
        analytics_store=analytics_store,
        analytics_warehouse=analytics_warehouse,
        event_bus=event_bus,
        event_bus_consumer=event_bus_consumer,
        hot_path_cache=hot_path_cache,
        thesis_vector_store=thesis_vector_store,
        feature_store=feature_store,
        backtesting_engine=backtesting_engine,
        semantic_analyst=semantic_analyst,
        evidently_observer=EvidentlyObserver(
            root_dir=settings.evidently_root_dir,
            enabled=settings.evidently_enabled,
            report_every_n_events=settings.evidently_report_every_n_events,
        ),
        braintrust_observer=BraintrustObserver(
            root_dir=settings.braintrust_root_dir,
            enabled=settings.braintrust_enabled,
            api_key=settings.braintrust_api_key,
            api_url=settings.braintrust_api_url,
            project_name=settings.braintrust_project_name,
            experiment_name=settings.braintrust_experiment_name,
            report_every_n_events=settings.braintrust_report_every_n_events,
        ),
        broker_client=broker_client,
        transcript_client=transcript_client,
        microstructure_client=microstructure_client,
        ownership_client=ownership_client,
        order_flow_client=order_flow_client,
        policy_feed_client=policy_feed_client,
        dbt_semantic_layer=dbt_semantic_layer,
        langfuse_observer=langfuse_observer,
        human_review_store=human_review_store,
        mlflow_observer=MLflowObserver(
            tracking_uri=settings.mlflow_tracking_uri,
            experiment_name=settings.mlflow_experiment_name,
        ),
    )
    ai_simulated_portfolio_service = AISimulatedPortfolioService(
        recommendation_service=recommendation_service,
        market_data_client=market_data_client,
        news_client=news_client,
        research_client=research_client,
        state_store=ai_simulated_portfolio_state_store,
        enabled=settings.ai_simulated_portfolio_enabled,
        starting_cash_usd=settings.ai_simulated_portfolio_starting_cash_usd,
        max_positions=settings.ai_simulated_portfolio_max_positions,
        max_position_pct=settings.ai_simulated_portfolio_max_position_pct,
        min_cash_pct=settings.ai_simulated_portfolio_min_cash_pct,
        min_trade_notional_usd=settings.ai_simulated_portfolio_min_trade_notional_usd,
        rebalance_interval_minutes=settings.ai_simulated_portfolio_rebalance_interval_minutes,
        core_tickers=tuple(
            item.strip().upper() for item in settings.ai_simulated_portfolio_core_tickers.split(",") if item.strip()
        ),
        profile_name=settings.ai_simulated_portfolio_profile,
        allowed_asset_types=tuple(
            item.strip().lower()
            for item in settings.ai_simulated_portfolio_allowed_asset_types.split(",")
            if item.strip()
        ),
        allow_fractional=settings.ai_simulated_portfolio_allow_fractional,
    )
    return create_application(
        bot_token=settings.telegram_token,
        recommendation_service=recommendation_service,
        market_data_client=market_data_client,
        news_client=news_client,
        research_client=research_client,
        broker_client=broker_client,
        transcript_client=transcript_client,
        microstructure_client=microstructure_client,
        live_market_stream_client=live_market_stream_client,
        workflow_orchestrator=workflow_orchestrator,
        market_news_limit=settings.market_news_limit,
        market_history_period=settings.market_history_period,
        market_history_interval=settings.market_history_interval,
        market_history_limit=settings.market_history_limit,
        telegram_report_chat_id=settings.telegram_report_chat_id,
        min_request_interval_seconds=settings.bot_min_request_interval_seconds,
        macro_event_refresh_interval_minutes=settings.macro_event_refresh_interval_minutes,
        macro_event_pre_window_minutes=settings.macro_event_pre_window_minutes,
        macro_event_post_window_minutes=settings.macro_event_post_window_minutes,
        macro_event_lookahead_hours=settings.macro_event_lookahead_hours,
        risk_vix_alert_threshold=settings.risk_vix_alert_threshold,
        risk_score_alert_threshold=settings.risk_score_alert_threshold,
        opportunity_score_alert_threshold=settings.opportunity_score_alert_threshold,
        news_impact_alert_threshold=settings.news_impact_alert_threshold,
        alert_state_store=alert_state_store,
        sector_rotation_state_store=sector_rotation_state_store,
        report_memory_store=report_memory_store,
        user_state_store=user_state_store,
        portfolio_state_store=portfolio_state_store,
        ai_simulated_portfolio_state_store=ai_simulated_portfolio_state_store,
        ai_simulated_portfolio_service=ai_simulated_portfolio_service,
        runtime_history_store=runtime_history_store,
        backup_manager=backup_manager,
        logs_dir=settings.logs_dir,
        log_retention=settings.log_retention,
        burn_in_target_days=settings.burn_in_target_days,
        health_alert_webhook_url=settings.health_alert_webhook_url,
        health_alert_webhook_secret=settings.health_alert_webhook_secret,
        health_alert_interval_minutes=settings.health_alert_interval_minutes,
        health_alert_cooldown_minutes=settings.health_alert_cooldown_minutes,
        health_alert_timeout_seconds=settings.health_alert_timeout_seconds,
        health_alert_retry_count=settings.health_alert_retry_count,
        health_alert_retry_backoff_seconds=settings.health_alert_retry_backoff_seconds,
        live_stream_symbols=tuple(item.strip().upper() for item in settings.live_stream_symbols.split(",") if item.strip()),
        live_stream_poll_interval_seconds=settings.live_stream_poll_interval_seconds,
        event_bus_consumer_poll_interval_seconds=settings.event_bus_consumer_poll_interval_seconds,
        live_stream_max_events=settings.live_stream_max_events,
        live_stream_spread_alert_bps=settings.live_stream_spread_alert_bps,
        telegram_transport=settings.telegram_transport,
        telegram_webhook_url=settings.telegram_webhook_url,
        telegram_webhook_path=settings.telegram_webhook_path,
        telegram_webhook_port=settings.telegram_webhook_port,
        jobs_enabled=settings.jobs_enabled,
        daily_digest_hour_utc=settings.daily_digest_hour_utc,
        daily_digest_minute_utc=settings.daily_digest_minute_utc,
        morning_report_hour_utc=settings.morning_report_hour_utc,
        morning_report_minute_utc=settings.morning_report_minute_utc,
        midday_report_hour_utc=settings.midday_report_hour_utc,
        midday_report_minute_utc=settings.midday_report_minute_utc,
        closing_report_hour_utc=settings.closing_report_hour_utc,
        closing_report_minute_utc=settings.closing_report_minute_utc,
        risk_check_interval_minutes=settings.risk_check_interval_minutes,
        ai_simulated_portfolio_rebalance_interval_minutes=settings.ai_simulated_portfolio_rebalance_interval_minutes,
        maintenance_cleanup_interval_minutes=settings.maintenance_cleanup_interval_minutes,
        earnings_alert_days_ahead=settings.earnings_alert_days_ahead,
        earnings_result_lookback_days=settings.earnings_result_lookback_days,
        sector_rotation_min_streak=settings.sector_rotation_min_streak,
        stock_pick_evaluation_horizon_days=settings.stock_pick_evaluation_horizon_days,
        database_url=effective_database_url,
        stock_pick_evaluation_interval_minutes=settings.stock_pick_evaluation_interval_minutes,
        backup_interval_hours=settings.backup_interval_hours,
    )


def resolve_database_url(settings: Settings) -> str:
    database_url = settings.database_url.strip()
    if not database_url:
        return ""
    try:
        importlib.import_module("psycopg")
    except ImportError as exc:
        logger.warning(
            "DATABASE_URL is configured but psycopg is unavailable; falling back to file-backed state: {}",
            exc,
        )
        return ""
    return database_url


def _build_health_details(application: object) -> dict[str, Any]:
    bot_data = getattr(application, "bot_data", {})
    if not isinstance(bot_data, dict):
        return {}
    services = bot_data.get("bot_services")
    snapshot = collect_runtime_snapshot(services)
    return {
        "mlflow": dict(snapshot.get("mlflow") or {}),
        "diagnostics": _build_diagnostics_payload(application),
    }


def _build_diagnostics_payload(application: object) -> dict[str, Any]:
    bot_data = getattr(application, "bot_data", {})
    services = bot_data.get("bot_services") if isinstance(bot_data, dict) else None
    return {
        "status": "ok",
        "service": "invest-advisor-bot",
        "runtime": collect_runtime_snapshot(services, ping_database=True),
    }


def _normalize_webhook_public_path(path: str) -> str:
    normalized = "/" + path.strip().strip("/")
    return normalized if normalized != "/" else "/telegram/webhook"


def _normalize_webhook_run_path(path: str) -> str:
    return _normalize_webhook_public_path(path).lstrip("/")


def _should_start_health_server(settings: Settings) -> bool:
    if not settings.health_check_enabled:
        return False
    if settings.telegram_transport != "webhook":
        return True
    return settings.health_check_port != settings.telegram_webhook_port


def _run_telegram_application(application: object, settings: Settings) -> None:
    if settings.telegram_transport == "webhook":
        webhook_public_path = _normalize_webhook_public_path(settings.telegram_webhook_path)
        webhook_url = settings.telegram_webhook_url.rstrip("/") + webhook_public_path
        run_webhook = getattr(application, "run_webhook")
        run_webhook(
            listen=settings.telegram_webhook_listen,
            port=settings.telegram_webhook_port,
            url_path=_normalize_webhook_run_path(settings.telegram_webhook_path),
            webhook_url=webhook_url,
            secret_token=settings.telegram_webhook_secret_token.strip() or None,
            drop_pending_updates=False,
        )
        return
    run_polling = getattr(application, "run_polling")
    run_polling(drop_pending_updates=False)


def main() -> int:
    try:
        settings = get_settings()
        settings.validate_runtime()
    except Exception as exc:
        print(f"Configuration error: {exc}", file=sys.stderr, flush=True)
        logger.error("Configuration error: {}", exc)
        return 2

    effective_database_url = resolve_database_url(settings)
    configure_logging(settings)
    log_event(
        "runtime_boot",
        database_backend="postgres" if effective_database_url else "file",
        jobs_enabled=settings.jobs_enabled,
        health_check_enabled=settings.health_check_enabled,
    )
    if not settings.llm_available():
        logger.warning("No LLM provider key is configured; the bot will use fallback summaries")
    if not settings.research_available():
        logger.warning("No research API key is configured; the bot will rely on RSS-only news context")
    if any(
        secret.strip()
        for secret in (
            settings.telegram_token,
            settings.llm_api_key,
            settings.gemini_api_key,
            settings.groq_api_key,
            settings.github_models_api_key,
            settings.tavily_api_key,
            settings.exa_api_key,
        )
    ):
        logger.warning("Rotate any token/API key that was previously shared or used for testing before production deploy")

    application = build_application(settings, database_url=effective_database_url)
    sync_service_diagnostics(application.bot_data.get("bot_services") if isinstance(getattr(application, "bot_data", None), dict) else None)
    logger.info("Starting Telegram investment advisor bot")

    if _should_start_health_server(settings):
        set_health_details_provider(lambda: _build_health_details(application))
        start_health_check_server(
            host=settings.health_check_host,
            port=settings.health_check_port,
        )
    elif settings.health_check_enabled and settings.telegram_transport == "webhook":
        logger.info(
            "Health check server skipped because TELEGRAM_TRANSPORT=webhook and HEALTH_CHECK_PORT matches TELEGRAM_WEBHOOK_PORT ({})",
            settings.telegram_webhook_port,
        )

    try:
        _run_telegram_application(application, settings)
    except KeyboardInterrupt:
        logger.info("Telegram bot stopped by user")
    except Exception as exc:
        logger.exception("Telegram bot crashed: {}", exc)
        return 1
    finally:
        with suppress(Exception):
            asyncio.run(_shutdown_clients(application))
        with suppress(Exception):
            stop_health_check_server()
        with suppress(Exception):
            logger.info("Telegram bot shutdown complete")

    return 0


async def _shutdown_clients(application: object) -> None:
    bot_data = getattr(application, "bot_data", {})
    if not isinstance(bot_data, dict):
        return
    for key in ("bot_services",):
        services = bot_data.get(key)
        if services is None:
            continue
        for attr_name in ("market_data_client", "news_client", "research_client", "recommendation_service", "live_market_stream_client"):
            instance = getattr(services, attr_name, None)
            if attr_name == "recommendation_service":
                instance = getattr(instance, "llm_client", None)
            aclose = getattr(instance, "aclose", None)
            if callable(aclose):
                await aclose()


if __name__ == "__main__":
    raise SystemExit(main())
