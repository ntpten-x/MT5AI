from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(slots=True, frozen=True)
class RankedAsset:
    asset: str
    label: str
    score: float
    stance: str
    rationale: str


def rank_asset_snapshots(
    asset_snapshots: Sequence[Mapping[str, object]],
    *,
    top_k: int = 5,
) -> list[RankedAsset]:
    ranked: list[RankedAsset] = []
    for snapshot in asset_snapshots:
        asset_name = str(snapshot.get("asset") or "")
        label = str(snapshot.get("label") or asset_name)
        trend = str(snapshot.get("trend") or "sideways")
        trend_score = _as_float(snapshot.get("trend_score")) or 0.0
        day_change_pct = _as_float(snapshot.get("day_change_pct")) or 0.0
        rsi = _as_float(snapshot.get("rsi")) or 50.0
        macd_hist = _as_float(snapshot.get("macd_hist")) or 0.0

        score = trend_score + (day_change_pct * 0.1) + (macd_hist * 0.5)
        if trend == "uptrend" and 50 <= rsi <= 68:
            score += 0.8
        if trend == "downtrend" and rsi <= 40:
            score -= 0.6

        if score >= 2.5:
            stance = "watch-long"
            rationale = "แนวโน้มและแรงส่งยังเด่นกว่ากลุ่ม"
        elif score <= -2.5:
            stance = "avoid"
            rationale = "แนวโน้มอ่อนแรงและยังไม่ควรเร่งเพิ่มน้ำหนัก"
        else:
            stance = "neutral"
            rationale = "สัญญาณยังไม่ชัดพอสำหรับการเพิ่มหรือลดน้ำหนักแรง"

        ranked.append(
            RankedAsset(
                asset=asset_name,
                label=label,
                score=round(score, 2),
                stance=stance,
                rationale=rationale,
            )
        )

    ranked.sort(key=lambda item: item.score, reverse=True)
    return ranked[: max(1, top_k)]


def _as_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
