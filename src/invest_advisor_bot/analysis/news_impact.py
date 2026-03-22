from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Mapping, Sequence

from invest_advisor_bot.providers.news_client import NewsArticle

_POSITIVE_KEYWORDS: tuple[str, ...] = (
    "rate cut",
    "cooling inflation",
    "soft landing",
    "stimulus",
    "record high",
    "breakout",
    "upgrade",
    "buyback",
    "surge",
)
_NEGATIVE_KEYWORDS: tuple[str, ...] = (
    "war",
    "recession",
    "crash",
    "default",
    "shutdown",
    "tariff",
    "inflation warning",
    "downgrade",
    "bankruptcy",
    "selloff",
    "sanction",
)
_DEFENSIVE_KEYWORDS: tuple[str, ...] = (
    "gold",
    "treasury",
    "bond",
    "safe haven",
    "volatility",
    "vix",
)


@dataclass(slots=True, frozen=True)
class NewsImpact:
    title: str
    source: str | None
    published_at: datetime | None
    sentiment: str
    impact_score: float
    related_bucket: str
    rationale: str


def score_news_impacts(
    articles: Sequence[NewsArticle],
    *,
    limit: int = 5,
    min_abs_score: float = 1.0,
) -> list[NewsImpact]:
    impacts: list[NewsImpact] = []
    for article in list(articles)[: max(limit, 0)]:
        title = article.title.strip()
        normalized = title.casefold()
        score = 0.0
        reasons: list[str] = []

        positive_hits = _collect_hits(normalized, _POSITIVE_KEYWORDS)
        negative_hits = _collect_hits(normalized, _NEGATIVE_KEYWORDS)
        defensive_hits = _collect_hits(normalized, _DEFENSIVE_KEYWORDS)

        if positive_hits:
            score += 2.0 + (0.5 * len(positive_hits))
            reasons.append(f"บวกจากคำหลัก {', '.join(positive_hits[:2])}")
        if negative_hits:
            score -= 2.5 + (0.5 * len(negative_hits))
            reasons.append(f"ลบจากคำหลัก {', '.join(negative_hits[:2])}")
        if defensive_hits:
            reasons.append(f"โยงกับสินทรัพย์ป้องกันความเสี่ยง เช่น {', '.join(defensive_hits[:2])}")

        if "fed" in normalized or "federal reserve" in normalized:
            score += 0.5 if score >= 0 else -0.5
            reasons.append("เกี่ยวข้องกับ Fed")
        if "cpi" in normalized or "inflation" in normalized:
            score += 0.5 if score >= 0 else -0.5
            reasons.append("เกี่ยวข้องกับเงินเฟ้อ")

        if abs(score) < min_abs_score:
            continue

        related_bucket = _detect_related_bucket(normalized)
        sentiment = "positive" if score > 0 else "negative"
        impacts.append(
            NewsImpact(
                title=title,
                source=article.source,
                published_at=article.published_at,
                sentiment=sentiment,
                impact_score=round(abs(score), 2),
                related_bucket=related_bucket,
                rationale=" | ".join(reasons) if reasons else "headline มีผลต่อภาวะตลาด",
            )
        )

    impacts.sort(key=lambda item: item.impact_score, reverse=True)
    return impacts


def summarize_news_bias(impacts: Iterable[NewsImpact]) -> Mapping[str, int]:
    positive = 0
    negative = 0
    for item in impacts:
        if item.sentiment == "positive":
            positive += 1
        else:
            negative += 1
    return {"positive": positive, "negative": negative}


def _collect_hits(text: str, keywords: Sequence[str]) -> list[str]:
    hits: list[str] = []
    for keyword in keywords:
        pattern = r"\b" + re.escape(keyword) + r"\b"
        if re.search(pattern, text):
            hits.append(keyword)
    return hits


def _detect_related_bucket(normalized_title: str) -> str:
    if any(keyword in normalized_title for keyword in ("gold", "bullion", "gld", "iau")):
        return "gold"
    if any(keyword in normalized_title for keyword in ("bond", "treasury", "yield", "tlt")):
        return "bonds"
    if any(keyword in normalized_title for keyword in ("etf", "s&p", "nasdaq", "equity", "stock", "spy", "qqq", "voo", "vti")):
        return "us_equity"
    return "macro"
