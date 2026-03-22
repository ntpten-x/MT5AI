"""External data providers for the investment advisor bot."""

from .llm_client import (
    DEFAULT_GEMINI_MODELS,
    DEFAULT_GITHUB_MODELS,
    DEFAULT_GROQ_MODELS,
    DEFAULT_OPENROUTER_FREE_MODELS,
    LLMProviderConfig,
    LLMTextResponse,
    OpenAICompatibleLLMClient,
    OpenAILLMClient,
    build_default_llm_client,
)
from .market_data_client import AssetQuote, EarningsEvent, MarketDataClient, OhlcvBar, StockFundamentals
from .news_client import NewsArticle, NewsClient
from .research_client import DEFAULT_RESEARCH_PROVIDER_ORDER, ResearchClient, ResearchFinding

__all__ = [
    "AssetQuote",
    "DEFAULT_GEMINI_MODELS",
    "DEFAULT_GITHUB_MODELS",
    "DEFAULT_GROQ_MODELS",
    "DEFAULT_OPENROUTER_FREE_MODELS",
    "DEFAULT_RESEARCH_PROVIDER_ORDER",
    "EarningsEvent",
    "LLMProviderConfig",
    "LLMTextResponse",
    "MarketDataClient",
    "NewsArticle",
    "NewsClient",
    "OpenAICompatibleLLMClient",
    "OpenAILLMClient",
    "OhlcvBar",
    "ResearchClient",
    "ResearchFinding",
    "StockFundamentals",
    "build_default_llm_client",
]
