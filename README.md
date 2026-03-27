# Invest Advisor Bot

Telegram investment advisor bot for continuous market monitoring, portfolio guidance, and alerting.

## Core Capabilities

- Pulls market prices and OHLCV with `yfinance`
- Supports optional `Polygon` low-latency routing for US stock quotes, aggregates, and options chain snapshots
- Pulls macro and market headlines from free RSS feeds
- Enriches macro context with `FRED`, `BLS`, `EIA`, and `Treasury Fiscal Data`
- Adds `BEA`, `ALFRED`, `CFTC COT`, Fed qualitative signals, and FINRA short-sale flow when available
- Adds optional `CME FedWatch` market-implied rate-path context and optional `Nasdaq Data Link` structured macro / factor datasets
- Adds optional `ECB`, `IMF`, and `World Bank` ex-US macro layers for cross-region regime context
- Adds optional `Trading Economics` ex-US economic calendar, forecasts, and prior/actual release context
- Adds optional `Financial Modeling Prep` earnings transcripts for management-tone / guidance commentary
- Adds optional `Databento` microstructure snapshots for equity / options depth signals
- Adds optional `Cboe Trade Alert` order-flow snapshots for options-flow sentiment, sweeps, and unusual activity
- Adds optional `Databento Live` stream polling for live trade/quote telemetry and spread alerts
- Adds optional `GDELT Context 2.0 / GEO 2.0` geopolitical and global-event enrichment inside macro intelligence
- Enriches company intelligence with `SEC EDGAR / data.sec.gov`
- Adds optional `SEC 13F + 13D/13G` ownership intelligence for activist / anchor-holder / watchlist-manager signals
- Tracks analyst ratings / target-price upside, insider Form 4 activity, and corporate actions inside company intelligence
- Adds ETF holdings / sector / country exposure summaries for broad-market and sector ETF context
- Computes `EMA`, `RSI`, `MACD`, support, and resistance
- Screens a large-cap US stock universe and ranks individual stocks
- Builds timed market reports for morning, midday, and closing windows
- Pulls basic fundamentals and valuation metrics with `yfinance`
- Uses `Alpha Vantage` and optional `Finnhub` fallbacks for earnings, expectations, and alternative signals
- Supports optional `OpenBB` fallback/provider wiring for quotes and history
- Supports optional `pgvector` thesis memory and optional `Qdrant` thesis-vector sync for external semantic retrieval
- Adds optional `Feast` feature-store scaffolding plus local feature snapshot logging for recommendation/outcome features
- Adds optional candidate `backtesting engine` with `vectorbt` detection and local backtest snapshots for stock screener context
- Adds optional `Great Expectations` validation gates before reasoning
- Adds optional `Evidently` local LLM evaluation datasets/reports
- Adds local `DuckDB + Parquet` analytics storage for recommendation/evaluation/runtime learning traces
- Adds optional `Redpanda/Kafka` event-bus publishing plus local JSONL fallback for recommendation/evaluation/market events
- Adds optional `event-bus consumer worker` for replaying Kafka/JSONL traffic back into warehouse/cache sinks
- Adds optional `Redis` hot-path cache and stream buffers plus local file fallback
- Adds optional `ClickHouse` real-time analytics warehouse plus local JSONL fallback tables
- Builds ClickHouse `materialized views` and daily topic/outcome rollups for operational analytics
- Adds optional `Alpaca` broker paper-trading sandbox with Telegram commands for account, buy, and sell testing
- Adds optional `Tradier` options/live execution provider support for brokerage and options routing
- Adds optional `Braintrust` production LLM evaluation logging with local JSONL fallbacks
- Adds official `Fed / ECB` speech and press-feed ingestion for policy-tone context
- Adds optional `Langfuse` tracing plus local human-review queue / completion workflow
- Adds optional `dbt Semantic Layer` scaffolding over analytics warehouse tables and metrics
- Adds optional `Prefect` workflow catalog for runtime snapshots, market updates, and broker polling orchestration
- Adds optional `semantic analyst` layer for natural-language questions over structured runtime and analytics data
- Fetches stock-specific news for shortlisted names
- Builds market regime and portfolio guidance for `conservative`, `balanced`, and `growth` investors
- Continuously scans news, ETFs, gold, and US equity proxies
- Sends Telegram alerts when it detects:
  - rising macro risk
  - high-impact headlines
  - standout assets worth watching
  - stock picks of the day and extra stock opportunities when multiple names stand out
  - sector rotation shifts and earnings calendar events
- Uses multi-provider LLM routing across Gemini, Groq, GitHub Models, OpenRouter, and OpenAI
- Can use free-capable providers first, then fall back automatically when quota is exhausted
- Can enrich market context with Tavily and Exa search
- Falls back to deterministic Thai analysis when LLM quota is unavailable

## Telegram Commands

- `/start`
- `/help`
- `/broker`
- `/paperbuy`
- `/papersell`
- `/profile`
- `/watchlist`
- `/market_update`
- `/analyst`
- `/reviewqueue`
- `/reviewdone`

## Local Run

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .[dev]
copy .env.example .env
python -m invest_advisor_bot.main
```

If you want MLflow tracing, install the optional extra before running:

```powershell
pip install -e .[dev,observability]
```

If you want Langfuse plus dbt semantic-layer support in the same environment:

```powershell
pip install -e .[dev,observability,semantic-layer]
```

If you want the full production data/evaluation stack enabled:

```powershell
pip install -r requirements.txt -e .[dev,observability,analytics,memory,quality,feature-store,evaluation,microstructure,streaming,backtesting,execution]
```

## Render Deployment

This repo includes [render.yaml](e:/AI%20Trade/MT5AI/render.yaml) for a Render web service deployment.

Recommended settings:

- keep `HEALTH_CHECK_ENABLED=true`
- set `JOBS_ENABLED=true`
- provide `TELEGRAM_TOKEN`
- provide `TELEGRAM_REPORT_CHAT_ID`
- for free-first mode provide one or more of `GEMINI_API_KEY`, `GROQ_API_KEY`, `CEREBRAS_API_KEY`, `CLOUDFLARE_API_TOKEN`, `HUGGINGFACE_API_KEY`, and `GITHUB_MODELS_API_KEY`
- optionally provide `OPENROUTER_API_KEY` and `LLM_API_KEY`
- for search enrichment provide `TAVILY_API_KEY` and `EXA_API_KEY`
- for market intelligence optionally provide `ALPHA_VANTAGE_API_KEY`, `FINNHUB_API_KEY`, `FRED_API_KEY`, `BLS_API_KEY`, `EIA_API_KEY`, `BEA_API_KEY`, `TRADING_ECONOMICS_API_KEY`
- optionally provide `POLYGON_API_KEY` for low-latency stock/option routing
- optionally provide `CME_FEDWATCH_API_URL` and `CME_FEDWATCH_API_KEY` for market-implied Fed path enrichment
- optionally provide `NASDAQ_DATA_LINK_API_KEY` plus `NASDAQ_DATA_LINK_DATASETS` for structured macro / factor datasets
- optionally configure `ECB_SERIES_MAP`, `IMF_API_BASE_URL` / `IMF_SERIES_MAP`, and `WORLD_BANK_INDICATOR_MAP` for ex-US macro regime enrichment
- optionally set `GLOBAL_MACRO_CALENDAR_COUNTRIES` and `GLOBAL_MACRO_CALENDAR_IMPORTANCE` to extend macro event calendars with ex-US releases from Trading Economics
- optionally provide `FMP_API_KEY` for earnings transcript / management commentary enrichment
- optionally provide `DATABENTO_API_KEY`, `DATABENTO_EQUITIES_DATASET`, and `DATABENTO_OPTIONS_DATASET` for microstructure depth snapshots
- optionally set `CBOE_TRADE_ALERT_ENABLED=true` with `CBOE_TRADE_ALERT_API_KEY` and `CBOE_TRADE_ALERT_BASE_URL` for options-flow analytics
- optionally provide `LIVE_STREAM_ENABLED=true` plus `LIVE_STREAM_DATASET` to enable Databento live stream polling and spread alerts
- optionally provide `GDELT_QUERY` to enrich macro intelligence with global event / geopolitical context
- optionally provide `BROKER_SANDBOX_ENABLED=true` with `ALPACA_API_KEY` and `ALPACA_API_SECRET` for paper-trading execution sandbox
- optionally set `BROKER_PROVIDER=tradier` with `TRADIER_ACCESS_TOKEN` and `TRADIER_ACCOUNT_ID` for Tradier execution / options routing
- optionally set `BRAINTRUST_ENABLED=true` with `BRAINTRUST_API_KEY` if you want production LLM quality logs uploaded to Braintrust
- optionally set `LANGFUSE_ENABLED=true` with `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` for Langfuse tracing plus local JSONL fallback
- optionally set `HUMAN_REVIEW_ENABLED=true` to queue fallback / low-confidence recommendations for manual review via `/reviewqueue` and `/reviewdone`
- optionally set `DBT_SEMANTIC_LAYER_ENABLED=true` if `dbt-core` is installed and you want semantic metrics scaffolding synced locally
- optionally set `PREFECT_ENABLED=true` if `prefect` is installed and you want flow orchestration
- optionally provide `OPENBB_PAT` and `OPENBB_BASE_URL` if you want OpenBB fallback routing
- optionally set `QDRANT_ENABLED=true` with `QDRANT_URL` if you want external thesis-vector memory alongside built-in pgvector/file memory
- optionally provide `THESIS_EMBEDDING_API_KEY` and keep `THESIS_EMBEDDING_MODEL=text-embedding-3-small` if you want real thesis embeddings plus rerank instead of heuristic vectors
- when thesis embeddings are enabled, keep `QDRANT_VECTOR_SIZE` aligned with the embedding dimensions you want stored, for example `256`
- optionally set `FEATURE_STORE_ENABLED=true` and `FEAST_ENABLED=true` to scaffold local Feast-compatible feature snapshots for recommendation/outcome learning
- optionally set `BACKTESTING_ENABLED=true` to enrich stock screening with candidate-vs-benchmark backtest snapshots
- optionally set `EVENT_BUS_ENABLED=true` with `EVENT_BUS_BROKERS` for Redpanda/Kafka event export
- optionally set `EVENT_BUS_CONSUMER_ENABLED=true` to run the replay consumer worker against Kafka or local JSONL event logs
- optionally set `HOT_PATH_CACHE_ENABLED=true` with `REDIS_URL` for low-latency cache/stream buffering
- optionally set `ANALYTICS_WAREHOUSE_ENABLED=true` with `CLICKHOUSE_URL` for real-time analytics warehousing
- optionally set `SEMANTIC_ANALYST_ENABLED=true` to expose `/analyst` and structured natural-language runtime queries
- provide `SEC_USER_AGENT` with a contact string for SEC EDGAR access
- optionally set `SEC_13F_MANAGER_CIKS` to a comma-separated manager watchlist for 13F look-through ownership context
- optionally provide `MLFLOW_TRACKING_URI` if you want recommendation/evaluation traces pushed to MLflow
  this also requires installing `mlflow` via `pip install -e .[observability]`
- set `DATA_QUALITY_ENABLED=true` and optionally `DATA_QUALITY_GX_ENABLED=true` if `great-expectations` is installed
- set `EVIDENTLY_ENABLED=true` if `evidently` is installed and you want local LLM evaluation reports
- keep `ANALYTICS_STORE_ENABLED=true` to persist recommendation/evaluation/runtime events into local DuckDB/Parquet

Start command:

```bash
python -m invest_advisor_bot.main
```

Health endpoint:

```text
/health
/diagnostics
/metrics
```

`/health` stays lightweight for platform probes. `/diagnostics` returns the structured runtime snapshot for external monitoring, including uptime, DB state, provider circuit state, MLflow state, and service/provider configuration summaries. `/metrics` exports the same runtime state in Prometheus text format.

Telegram operators can also use `/status`, `/dashboard`, `/analyst`, `/reviewqueue`, and `/reviewdone` to inspect the streaming, cache, warehouse, semantic analyst, and human-review stack from the report chat. The bot also exposes `/ai_portfolio`, `/ai_trades`, `/ai_performance`, `/ai_rebalance`, and `/ai_reset` for the built-in simulated AI portfolio. Use `/ai_rebalance conservative|balanced|growth` or `/ai_reset 1000 balanced` to switch policy. On Render with Telegram polling, keep only one running bot instance or Telegram will terminate the duplicate `getUpdates` session.

Recommended Render transport:

```env
TELEGRAM_TRANSPORT=webhook
TELEGRAM_WEBHOOK_URL=https://your-service.onrender.com
TELEGRAM_WEBHOOK_PATH=/telegram/webhook
TELEGRAM_WEBHOOK_SECRET_TOKEN=replace-with-random-secret
TELEGRAM_WEBHOOK_LISTEN=0.0.0.0
TELEGRAM_WEBHOOK_PORT=10000
```

Go-live hardening checklist:

1. Rotate every token and API key that was previously shared in chat or logs.
2. Use `TELEGRAM_TRANSPORT=webhook` on Render so only one public bot instance receives updates.
3. Keep exactly one live deployment for the Telegram bot token; no local polling process should run in parallel.
4. Set `TELEGRAM_WEBHOOK_SECRET_TOKEN` to a random secret before production.
5. Verify `/status`, `/ai_portfolio`, `/market_update`, and one scheduled report after deploy.
6. Watch provider rate-limit warnings for 3-7 days before changing policy thresholds.

## Important Environment Variables

```env
TELEGRAM_TOKEN=
TELEGRAM_REPORT_CHAT_ID=
LLM_PROVIDER=auto
LLM_PROVIDER_ORDER=gemini,groq,cerebras,cloudflare,openrouter,huggingface,github_models,openai
GEMINI_API_KEY=
GROQ_API_KEY=
GITHUB_MODELS_API_KEY=
CEREBRAS_API_KEY=
CEREBRAS_MODELS=gpt-oss-120b,zai-glm-4.7
CEREBRAS_BASE_URL=https://api.cerebras.ai/v1
CLOUDFLARE_ACCOUNT_ID=
CLOUDFLARE_API_TOKEN=
CLOUDFLARE_MODELS=@cf/meta/llama-3.1-8b-instruct-fast,@cf/meta/llama-3.1-8b-instruct
CLOUDFLARE_BASE_URL=
HUGGINGFACE_API_KEY=
HUGGINGFACE_MODELS=google/gemma-2-2b-it,Qwen/Qwen2.5-7B-Instruct-1M
HUGGINGFACE_BASE_URL=https://router.huggingface.co/v1
OPENROUTER_API_KEY=
LLM_API_KEY=
TAVILY_API_KEY=
EXA_API_KEY=
ALPHA_VANTAGE_API_KEY=
FINNHUB_API_KEY=
FRED_API_KEY=
BLS_API_KEY=
EIA_API_KEY=
BEA_API_KEY=
OPENBB_PAT=
OPENBB_BASE_URL=
AI_SIM_PORTFOLIO_ENABLED=true
AI_SIM_PORTFOLIO_STATE_PATH=data/ai_simulated_portfolio.json
AI_SIM_PORTFOLIO_STARTING_CASH_USD=1000
AI_SIM_PORTFOLIO_MAX_POSITIONS=5
AI_SIM_PORTFOLIO_MAX_POSITION_PCT=0.25
AI_SIM_PORTFOLIO_MIN_CASH_PCT=0.10
AI_SIM_PORTFOLIO_MIN_TRADE_NOTIONAL_USD=25
AI_SIM_PORTFOLIO_REBALANCE_INTERVAL_MINUTES=360
AI_SIM_PORTFOLIO_CORE_TICKERS=SPY,QQQ,VTI,VOO,GLD,IAU,TLT
AI_SIM_PORTFOLIO_PROFILE=growth
AI_SIM_PORTFOLIO_ALLOWED_ASSET_TYPES=stock,etf,gold
AI_SIM_PORTFOLIO_ALLOW_FRACTIONAL=true
QDRANT_ENABLED=false
QDRANT_URL=
QDRANT_API_KEY=
QDRANT_COLLECTION_NAME=invest_advisor_thesis_memory
QDRANT_VECTOR_SIZE=256
THESIS_EMBEDDING_API_KEY=
THESIS_EMBEDDING_BASE_URL=https://api.openai.com/v1
THESIS_EMBEDDING_MODEL=text-embedding-3-small
THESIS_EMBEDDING_TIMEOUT_SECONDS=12
THESIS_RERANK_ENABLED=true
FEATURE_STORE_ENABLED=false
FEAST_ENABLED=false
FEAST_PROJECT_NAME=invest_advisor_bot
BACKTESTING_ENABLED=false
BACKTESTING_BENCHMARK_TICKER=SPY
BACKTESTING_LOOKBACK_PERIOD=6mo
TRADING_ECONOMICS_API_KEY=
POLYGON_API_KEY=
POLYGON_BASE_URL=https://api.polygon.io
CME_FEDWATCH_API_KEY=
CME_FEDWATCH_API_URL=
NASDAQ_DATA_LINK_API_KEY=
NASDAQ_DATA_LINK_DATASETS=
ECB_SERIES_MAP=inflation_yoy=ICP/M.U2.N.000000.4.ANR
IMF_API_BASE_URL=
IMF_SERIES_MAP=
WORLD_BANK_COUNTRIES=EMU,EUU,JPN,CHN
WORLD_BANK_INDICATOR_MAP=gdp_growth=NY.GDP.MKTP.KD.ZG,unemployment=SL.UEM.TOTL.ZS
GLOBAL_MACRO_CALENDAR_COUNTRIES=Euro Area,Japan,China,United Kingdom,Canada,Australia
GLOBAL_MACRO_CALENDAR_IMPORTANCE=2
DATABENTO_API_KEY=
DATABENTO_EQUITIES_DATASET=
DATABENTO_OPTIONS_DATASET=
CBOE_TRADE_ALERT_ENABLED=false
CBOE_TRADE_ALERT_API_KEY=
CBOE_TRADE_ALERT_BASE_URL=
ORDER_FLOW_TIMEOUT_SECONDS=12
LIVE_STREAM_ENABLED=false
LIVE_STREAM_DATASET=
GDELT_QUERY=
FMP_API_KEY=
BROKER_SANDBOX_ENABLED=false
ALPACA_API_KEY=
ALPACA_API_SECRET=
BROKER_PROVIDER=alpaca
TRADIER_ACCESS_TOKEN=
TRADIER_ACCOUNT_ID=
TRADIER_BASE_URL=https://sandbox.tradier.com
BRAINTRUST_ENABLED=false
BRAINTRUST_API_KEY=
BRAINTRUST_API_URL=https://api.braintrust.dev
LANGFUSE_ENABLED=false
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_HOST=https://cloud.langfuse.com
HUMAN_REVIEW_ENABLED=false
HUMAN_REVIEW_LOW_CONFIDENCE_THRESHOLD=0.58
HUMAN_REVIEW_SAMPLE_EVERY_N=5
DBT_SEMANTIC_LAYER_ENABLED=false
DBT_SEMANTIC_PROJECT_NAME=invest_advisor_bot_semantic
DBT_SEMANTIC_TARGET_SCHEMA=analytics
PREFECT_ENABLED=false
SEC_USER_AGENT=InvestAdvisorBot/0.2 support@example.com
SEC_13F_MANAGER_CIKS=
PGVECTOR_ENABLED=true
THESIS_MEMORY_TOP_K=3
MLFLOW_TRACKING_URI=
MLFLOW_EXPERIMENT_NAME=invest-advisor-bot
DATA_QUALITY_ENABLED=true
DATA_QUALITY_GX_ENABLED=false
ANALYTICS_STORE_ENABLED=true
EVENT_BUS_ENABLED=false
EVENT_BUS_BROKERS=
EVENT_BUS_CONSUMER_ENABLED=false
EVENT_BUS_CONSUMER_GROUP=invest-advisor-bot
HOT_PATH_CACHE_ENABLED=false
REDIS_URL=
ANALYTICS_WAREHOUSE_ENABLED=false
CLICKHOUSE_URL=
SEMANTIC_ANALYST_ENABLED=false
EVIDENTLY_ENABLED=false
HEALTH_ALERT_WEBHOOK_URL=
HEALTH_ALERT_WEBHOOK_SECRET=
HEALTH_ALERT_INTERVAL_MINUTES=5
HEALTH_ALERT_COOLDOWN_MINUTES=30
HEALTH_ALERT_TIMEOUT_SECONDS=8
HEALTH_ALERT_RETRY_COUNT=3
HEALTH_ALERT_RETRY_BACKOFF_SECONDS=1.5
DEFAULT_INVESTOR_PROFILE=conservative
JOBS_ENABLED=true
RISK_CHECK_INTERVAL_MINUTES=30
MACRO_EVENT_REFRESH_INTERVAL_MINUTES=5
MACRO_EVENT_PRE_WINDOW_MINUTES=20
MACRO_EVENT_POST_WINDOW_MINUTES=90
MACRO_EVENT_LOOKAHEAD_HOURS=12
ALERT_STATE_PATH=data/alert_state.json
USER_STATE_PATH=data/user_state.json
ALERT_SUPPRESSION_MINUTES=180
RISK_SCORE_ALERT_THRESHOLD=6.5
OPPORTUNITY_SCORE_ALERT_THRESHOLD=2.8
NEWS_IMPACT_ALERT_THRESHOLD=2.0
MORNING_REPORT_HOUR_UTC=12
MIDDAY_REPORT_HOUR_UTC=17
CLOSING_REPORT_HOUR_UTC=21
EARNINGS_ALERT_DAYS_AHEAD=7
```

## Notes

- In `LLM_PROVIDER=auto`, the bot follows `LLM_PROVIDER_ORDER` and skips providers whose keys are missing.
- Gemini, OpenAI, and OpenRouter quotas can be exhausted independently; the router keeps moving until a provider succeeds.
- If the LLM key has no quota, the bot still runs with deterministic fallback summaries.
- MLflow tracing stays disabled unless `mlflow` is installed; setting `MLFLOW_TRACKING_URI` alone is not enough.
- `Polygon`, `CME FedWatch`, and `Nasdaq Data Link` are optional enrichments. The bot still runs without them, but the low-latency path, options snapshots, implied macro layer, and structured macro datasets will stay empty.
- `Trading Economics` powers both US consensus surprises and optional ex-US macro calendar events. If `GLOBAL_MACRO_CALENDAR_COUNTRIES` is set, the payload will include country, forecast, previous, and actual values when available.
- ETF holdings / exposure, analyst ratings, and corporate actions currently use `yfinance` / `Finnhub` / `SEC` fallbacks; keep `SEC_USER_AGENT` configured for production-quality insider coverage.
- `ECB`, `IMF`, and `World Bank` enrich ex-US macro context. Without those mappings or endpoints, macro intelligence remains US-centric.
- `Financial Modeling Prep` earnings transcripts are optional. Without `FMP_API_KEY`, management-tone / guidance commentary stays empty.
- `Databento` is optional. Without it, microstructure and options-depth signals stay empty even though the rest of the bot still runs.
- `Databento Live` is optional. When enabled, the scheduled stream sampler records live trade/quote events into analytics storage and can raise spread alerts in the report chat.
- `GDELT Context 2.0 / GEO 2.0` are optional. Without `GDELT_QUERY`, the geopolitical/global-event layer is skipped and macro intelligence stays focused on market/macro data only.
- `Qdrant` is optional. When enabled, the bot mirrors thesis memory into an external vector collection while keeping local JSONL fallbacks.
- For higher-quality thesis recall, set `THESIS_EMBEDDING_API_KEY`, keep `THESIS_EMBEDDING_MODEL=text-embedding-3-small`, and align `QDRANT_VECTOR_SIZE` with the embedding dimensions you want stored.
- `Feast` is optional. The bot always writes local feature snapshots and, when `FEAST_ENABLED=true`, scaffolds a local Feast-compatible repo for recommendation/outcome features.
- `Backtesting` is optional. When enabled, the stock screener includes recent candidate-vs-benchmark backtest context and persists local backtest summaries.
- `Alpaca` broker paper trading stays disabled unless `BROKER_SANDBOX_ENABLED=true` and valid paper credentials are configured. `/broker`, `/paperbuy`, and `/papersell` are intended for sandbox use only.
- `Tradier` can be used instead of Alpaca by setting `BROKER_PROVIDER=tradier`. The bot will use Tradier account balances/positions/orders and can submit option orders through the execution client.
- `Braintrust` is optional. If the SDK or API key is unavailable, the observer still keeps local JSONL datasets for recommendation/outcome quality events and marks remote sync as unavailable.
- `Prefect` is optional. If the dependency is missing or `PREFECT_ENABLED=false`, the workflow catalog still reports status but orchestration wrappers fall back to direct function calls.
- `DATA_QUALITY_GX_ENABLED=true` requires `great-expectations`. If the dependency is missing, the built-in guard still runs and the GX portion of the report is marked unavailable.
- `EVIDENTLY_ENABLED=true` requires `evidently`. Recommendation-time rows and realized outcome rows are written to the local Evidently dataset folder, and HTML/JSON reports are generated on the configured cadence when the package is present.
- `ANALYTICS_STORE_ENABLED=true` writes recommendation/evaluation/runtime events into `DuckDB` when available and always keeps JSONL fallbacks; periodic Parquet exports are refreshed in the analytics folder.
- `EVENT_BUS_ENABLED=true` publishes recommendation/evaluation/market events into Kafka-compatible brokers such as Redpanda. If the broker dependency or connection is unavailable, local JSONL fallbacks still persist the event stream.
- `EVENT_BUS_CONSUMER_ENABLED=true` starts the replay consumer path. It tracks JSONL byte offsets locally and can also read Kafka topics under the configured prefix, then fan those events back into the warehouse/cache sinks.
- `HOT_PATH_CACHE_ENABLED=true` enables Redis-backed hot cache and stream buffers. If Redis is unavailable, the bot falls back to local in-memory/file storage for operator inspection and replay.
- `ANALYTICS_WAREHOUSE_ENABLED=true` writes high-frequency events into ClickHouse when configured and keeps per-table JSONL fallbacks locally for recovery or offline replay. The warehouse also provisions materialized views and rollup tables such as recommendation topic, evaluation outcome, market topic, and runtime topic rollups.
- `SEMANTIC_ANALYST_ENABLED=true` unlocks `/analyst`. Without a remote analyst endpoint, the built-in local heuristic mode still answers operational questions from warehouse snapshots and recent events.
- `HEALTH_ALERT_WEBHOOK_URL` enables webhook alerts for runtime incidents such as unhealthy DB, open provider circuits, or active MLflow warnings. The monitor only re-sends unchanged incidents after the configured cooldown and emits a resolved event when the runtime recovers.
- `HEALTH_ALERT_WEBHOOK_SECRET` adds `X-Invest-Advisor-Timestamp` and `X-Invest-Advisor-Signature` HMAC headers so downstream alert routers can verify authenticity. Every delivery also includes `X-Invest-Advisor-Idempotency-Key` plus `incident_key` and `idempotency_key` in the JSON body. `HEALTH_ALERT_RETRY_COUNT` and `HEALTH_ALERT_RETRY_BACKOFF_SECONDS` control retry/backoff for transient webhook delivery failures.
- `/metrics` now exposes labeled provider latency metrics such as `service`, `provider`, and `operation`, including rolling `p95` and `p99`, so Prometheus can track source/provider latency SLOs directly.
- SEC EDGAR requests should send a real contact-style `SEC_USER_AGENT` before production use.
- Alert state is persisted to disk to reduce duplicate Telegram spam across restarts.
- User watchlists and alert preferences are persisted to disk.
- Continuous monitoring depends on the process staying online. Use an always-on Render service plan if you need uninterrupted scanning.

## Development

```powershell
pytest
python -m compileall src\invest_advisor_bot tests
```
