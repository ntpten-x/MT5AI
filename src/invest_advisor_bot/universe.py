from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class StockUniverseMember:
    ticker: str
    company_name: str
    sector: str
    benchmark: str


INDEX_UNIVERSE_SOURCES: dict[str, dict[str, str]] = {
    "sp500": {
        "url": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        "ticker_column": "Symbol",
        "name_column": "Security",
        "sector_column": "GICS Sector",
        "table_hint": "Symbol",
    },
    "nasdaq100": {
        "url": "https://en.wikipedia.org/wiki/Nasdaq-100",
        "ticker_column": "Ticker",
        "name_column": "Company",
        "sector_column": "GICS Sector",
        "table_hint": "Ticker",
    },
}


US_LARGE_CAP_STOCK_UNIVERSE: dict[str, StockUniverseMember] = {
    "aapl": StockUniverseMember("AAPL", "Apple", "Technology", "nasdaq100"),
    "msft": StockUniverseMember("MSFT", "Microsoft", "Technology", "nasdaq100"),
    "nvda": StockUniverseMember("NVDA", "NVIDIA", "Technology", "nasdaq100"),
    "amzn": StockUniverseMember("AMZN", "Amazon", "Consumer Discretionary", "nasdaq100"),
    "meta": StockUniverseMember("META", "Meta Platforms", "Communication Services", "nasdaq100"),
    "googl": StockUniverseMember("GOOGL", "Alphabet Class A", "Communication Services", "nasdaq100"),
    "goog": StockUniverseMember("GOOG", "Alphabet Class C", "Communication Services", "nasdaq100"),
    "tsla": StockUniverseMember("TSLA", "Tesla", "Consumer Discretionary", "nasdaq100"),
    "avgo": StockUniverseMember("AVGO", "Broadcom", "Technology", "nasdaq100"),
    "cost": StockUniverseMember("COST", "Costco", "Consumer Staples", "nasdaq100"),
    "nflx": StockUniverseMember("NFLX", "Netflix", "Communication Services", "nasdaq100"),
    "amd": StockUniverseMember("AMD", "Advanced Micro Devices", "Technology", "nasdaq100"),
    "adbe": StockUniverseMember("ADBE", "Adobe", "Technology", "nasdaq100"),
    "intc": StockUniverseMember("INTC", "Intel", "Technology", "nasdaq100"),
    "intu": StockUniverseMember("INTU", "Intuit", "Technology", "nasdaq100"),
    "amgn": StockUniverseMember("AMGN", "Amgen", "Healthcare", "nasdaq100"),
    "isrg": StockUniverseMember("ISRG", "Intuitive Surgical", "Healthcare", "nasdaq100"),
    "bkng": StockUniverseMember("BKNG", "Booking Holdings", "Consumer Discretionary", "nasdaq100"),
    "qcom": StockUniverseMember("QCOM", "Qualcomm", "Technology", "nasdaq100"),
    "txn": StockUniverseMember("TXN", "Texas Instruments", "Technology", "nasdaq100"),
    "orcl": StockUniverseMember("ORCL", "Oracle", "Technology", "sp500"),
    "crm": StockUniverseMember("CRM", "Salesforce", "Technology", "sp500"),
    "jpm": StockUniverseMember("JPM", "JPMorgan Chase", "Financials", "sp500"),
    "brk_b": StockUniverseMember("BRK-B", "Berkshire Hathaway", "Financials", "sp500"),
    "v": StockUniverseMember("V", "Visa", "Financials", "sp500"),
    "ma": StockUniverseMember("MA", "Mastercard", "Financials", "sp500"),
    "unh": StockUniverseMember("UNH", "UnitedHealth", "Healthcare", "sp500"),
    "lly": StockUniverseMember("LLY", "Eli Lilly", "Healthcare", "sp500"),
    "jnj": StockUniverseMember("JNJ", "Johnson & Johnson", "Healthcare", "sp500"),
    "abbv": StockUniverseMember("ABBV", "AbbVie", "Healthcare", "sp500"),
    "xom": StockUniverseMember("XOM", "Exxon Mobil", "Energy", "sp500"),
    "cvx": StockUniverseMember("CVX", "Chevron", "Energy", "sp500"),
    "cat": StockUniverseMember("CAT", "Caterpillar", "Industrials", "sp500"),
    "lin": StockUniverseMember("LIN", "Linde", "Materials", "sp500"),
    "pg": StockUniverseMember("PG", "Procter & Gamble", "Consumer Staples", "sp500"),
    "ko": StockUniverseMember("KO", "Coca-Cola", "Consumer Staples", "sp500"),
    "pep": StockUniverseMember("PEP", "PepsiCo", "Consumer Staples", "sp500"),
    "wmt": StockUniverseMember("WMT", "Walmart", "Consumer Staples", "sp500"),
    "hd": StockUniverseMember("HD", "Home Depot", "Consumer Discretionary", "sp500"),
    "mcd": StockUniverseMember("MCD", "McDonald's", "Consumer Discretionary", "sp500"),
    "dis": StockUniverseMember("DIS", "Walt Disney", "Communication Services", "sp500"),
    "ge": StockUniverseMember("GE", "GE Aerospace", "Industrials", "sp500"),
    "uber": StockUniverseMember("UBER", "Uber Technologies", "Industrials", "sp500"),
}


def find_stock_candidates_from_text(text: str) -> list[StockUniverseMember]:
    normalized = text.casefold()
    matches: list[StockUniverseMember] = []
    for alias, member in US_LARGE_CAP_STOCK_UNIVERSE.items():
        if alias.casefold() in normalized or member.ticker.casefold() in normalized or member.company_name.casefold() in normalized:
            matches.append(member)
    return matches


def normalize_ticker_for_market_data(ticker: str) -> str:
    return ticker.strip().upper().replace(".", "-")
