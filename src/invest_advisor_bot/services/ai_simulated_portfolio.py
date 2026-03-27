from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping, Sequence

from invest_advisor_bot.analysis.portfolio_profile import normalize_profile_name
from invest_advisor_bot.bot.ai_simulated_portfolio_state import (
    AISimulatedPortfolioHolding,
    AISimulatedPortfolioState,
    AISimulatedPortfolioStateStore,
    AISimulatedPortfolioTrade,
)
from invest_advisor_bot.bot.portfolio_state import PortfolioHolding
from invest_advisor_bot.observability import log_event
from invest_advisor_bot.providers.market_data_client import AssetQuote, MarketDataClient
from invest_advisor_bot.providers.news_client import NewsClient
from invest_advisor_bot.providers.research_client import ResearchClient
from invest_advisor_bot.services.recommendation_service import RecommendationService


@dataclass(slots=True, frozen=True)
class AISimulatedPolicy:
    name: str
    max_positions: int
    max_position_pct: float
    min_cash_pct: float
    min_market_confidence: float
    min_stock_confidence: float
    min_coverage_score: float
    min_hold_days: int
    reentry_cooldown_days: int
    max_turnover_pct: float


PROFILE_POLICIES: dict[str, AISimulatedPolicy] = {
    "conservative": AISimulatedPolicy("conservative", 3, 0.22, 0.25, 0.58, 0.72, 0.62, 5, 4, 0.25),
    "balanced": AISimulatedPolicy("balanced", 4, 0.27, 0.15, 0.50, 0.64, 0.54, 3, 3, 0.35),
    "growth": AISimulatedPolicy("growth", 5, 0.30, 0.08, 0.42, 0.58, 0.45, 1, 2, 0.45),
}

RISK_OFF_SAFE_TICKERS = {"GLD", "IAU", "TLT"}


@dataclass(slots=True, frozen=True)
class AISimulatedPortfolioRebalanceResult:
    state: AISimulatedPortfolioState
    snapshot: dict[str, Any]
    decisions: tuple[dict[str, Any], ...]
    trades: tuple[AISimulatedPortfolioTrade, ...]
    action_count: int
    rendered_summary: str
    market_payload: dict[str, Any]
    skipped_reason: str | None = None


class AISimulatedPortfolioService:
    """Bot-owned long-only paper portfolio for example allocations and alerts."""

    def __init__(
        self,
        *,
        recommendation_service: RecommendationService,
        market_data_client: MarketDataClient,
        news_client: NewsClient,
        research_client: ResearchClient | None,
        state_store: AISimulatedPortfolioStateStore,
        enabled: bool = True,
        starting_cash_usd: float = 1000.0,
        max_positions: int = 5,
        max_position_pct: float = 0.25,
        min_cash_pct: float = 0.10,
        min_trade_notional_usd: float = 25.0,
        rebalance_interval_minutes: int = 360,
        core_tickers: Sequence[str] = ("SPY", "QQQ", "VTI", "VOO", "GLD", "IAU", "TLT"),
        allow_fractional: bool = True,
        profile_name: str = "growth",
        allowed_asset_types: Sequence[str] = ("stock", "etf", "gold"),
    ) -> None:
        self.recommendation_service = recommendation_service
        self.market_data_client = market_data_client
        self.news_client = news_client
        self.research_client = research_client
        self.state_store = state_store
        self.enabled = enabled
        self.starting_cash_usd = max(100.0, float(starting_cash_usd))
        self.default_max_positions = max(1, int(max_positions))
        self.default_max_position_pct = min(max(float(max_position_pct), 0.05), 0.50)
        self.default_min_cash_pct = min(max(float(min_cash_pct), 0.0), 0.50)
        self.min_trade_notional_usd = max(5.0, float(min_trade_notional_usd))
        self.rebalance_interval_minutes = max(30, int(rebalance_interval_minutes))
        self.core_tickers = tuple(dict.fromkeys(item.strip().upper() for item in core_tickers if item.strip()))
        self.allow_fractional = allow_fractional
        self.default_profile_name = self._normalize_profile_name(profile_name)
        normalized_asset_types = tuple(
            dict.fromkeys(item.strip().lower() for item in allowed_asset_types if item and item.strip())
        )
        self.allowed_asset_types = normalized_asset_types or ("stock", "etf", "gold")

    def portfolio_key(self, conversation_key: str | None) -> str:
        normalized = str(conversation_key or "").strip()
        return normalized or "ai-simulated-portfolio"

    def ensure_portfolio(self, conversation_key: str | None) -> AISimulatedPortfolioState:
        state = self.state_store.ensure_portfolio(
            self.portfolio_key(conversation_key),
            starting_cash=self.starting_cash_usd,
            base_currency="USD",
            metadata={
                "engine": "ai-simulated-portfolio",
                "profile_name": self.default_profile_name,
                "allowed_asset_types": list(self.allowed_asset_types),
                "realized_pnl_by_ticker": {},
                "last_exit_at_by_ticker": {},
                "labels_by_ticker": {},
            },
        )
        metadata = self._normalized_metadata(state)
        if metadata != state.metadata:
            state = self.state_store.save_portfolio(
                self.portfolio_key(conversation_key),
                starting_cash=state.starting_cash,
                cash=state.cash,
                realized_pnl=state.realized_pnl,
                holdings=[self._serialize_holding_payload(item) for item in state.holdings],
                last_rebalanced_at=state.last_rebalanced_at,
                last_action_summary=state.last_action_summary,
                metadata=metadata,
            )
        return state

    def set_profile(self, *, conversation_key: str | None, profile_name: str) -> AISimulatedPortfolioState:
        state = self.ensure_portfolio(conversation_key)
        metadata = self._normalized_metadata(state)
        metadata["profile_name"] = self._normalize_profile_name(profile_name)
        return self.state_store.save_portfolio(
            self.portfolio_key(conversation_key),
            starting_cash=state.starting_cash,
            cash=state.cash,
            realized_pnl=state.realized_pnl,
            holdings=[self._serialize_holding_payload(item) for item in state.holdings],
            last_rebalanced_at=state.last_rebalanced_at,
            last_action_summary=state.last_action_summary,
            metadata=metadata,
        )

    async def maybe_rebalance(
        self,
        *,
        conversation_key: str | None,
        reason: str,
        force: bool = False,
    ) -> AISimulatedPortfolioRebalanceResult:
        state = self.ensure_portfolio(conversation_key)
        if not self.enabled:
            snapshot = await self.build_snapshot(conversation_key=conversation_key)
            return AISimulatedPortfolioRebalanceResult(state, snapshot, (), (), 0, "AI Portfolio: disabled", {}, "disabled")
        if not force and state.last_rebalanced_at is not None:
            staleness = datetime.now(timezone.utc) - state.last_rebalanced_at
            if staleness < timedelta(minutes=self.rebalance_interval_minutes):
                snapshot = await self.build_snapshot(conversation_key=conversation_key)
                return AISimulatedPortfolioRebalanceResult(
                    state,
                    snapshot,
                    (),
                    (),
                    0,
                    self.render_snapshot_text(snapshot),
                    {},
                    "cooldown_active",
                )

        market_result = await self.recommendation_service.generate_market_update(
            news_client=self.news_client,
            market_data_client=self.market_data_client,
            research_client=self.research_client,
            news_limit=5,
            history_period="6mo",
            history_interval="1d",
            history_limit=180,
            conversation_key=self.portfolio_key(conversation_key),
            portfolio_holdings=self._state_to_portfolio_holdings(state),
        )
        payload = dict(market_result.input_payload or {})
        policy = self._resolve_policy(state)
        market_confidence = self._market_confidence_score(payload)
        abstain_mode = market_confidence < policy.min_market_confidence
        deep_risk_off = market_confidence < max(0.18, policy.min_market_confidence - 0.12)
        candidates = self._select_candidates(payload, policy=policy, abstain_mode=abstain_mode)
        all_tickers = {holding.normalized_ticker for holding in state.holdings}
        all_tickers.update(str(item.get("ticker") or "").upper() for item in candidates if item.get("ticker"))
        quotes = await self.market_data_client.get_latest_prices(sorted(ticker for ticker in all_tickers if ticker))
        valuation = self._compute_valuation(state=state, quotes=quotes)
        total_value = valuation["total_value"]
        cash_floor_value = total_value * policy.min_cash_pct
        target_value_by_ticker = {
            ticker: total_value * weight
            for ticker, weight in self._build_target_weights(
                candidates=candidates,
                policy=policy,
                abstain_mode=abstain_mode,
                deep_risk_off=deep_risk_off,
            ).items()
        }
        return await self._execute_rebalance(
            conversation_key=conversation_key,
            reason=reason,
            state=state,
            payload=payload,
            policy=policy,
            abstain_mode=abstain_mode,
            deep_risk_off=deep_risk_off,
            candidates=candidates,
            quotes=quotes,
            target_value_by_ticker=target_value_by_ticker,
            total_value=total_value,
            cash_floor_value=cash_floor_value,
        )

    async def build_snapshot(self, *, conversation_key: str | None) -> dict[str, Any]:
        state = self.ensure_portfolio(conversation_key)
        metadata = self._normalized_metadata(state)
        tickers = sorted({holding.normalized_ticker for holding in state.holdings})
        quotes = await self.market_data_client.get_latest_prices(tickers) if tickers else {}
        valuation = self._compute_valuation(state=state, quotes=quotes)
        total_value = valuation["total_value"]
        holdings: list[dict[str, Any]] = []
        labels_by_ticker = dict(metadata.get("labels_by_ticker") or {})
        for holding in state.holdings:
            ticker = holding.normalized_ticker
            quote = quotes.get(ticker)
            price = self._coerce_float(quote.price if quote is not None else None)
            if price is None or price <= 0:
                price = self._coerce_float(holding.avg_cost) or 0.0
            market_value = holding.quantity * price
            avg_cost = self._coerce_float(holding.avg_cost)
            unrealized_pnl = (price - avg_cost) * holding.quantity if avg_cost is not None and price > 0 else 0.0
            return_pct = ((price - avg_cost) / avg_cost) if avg_cost not in (None, 0.0) and price > 0 else None
            opened_at = holding.opened_at
            holdings.append(
                {
                    "ticker": ticker,
                    "label": holding.label or str(labels_by_ticker.get(ticker) or ticker),
                    "asset_type": holding.asset_type,
                    "quantity": holding.quantity,
                    "avg_cost": avg_cost,
                    "price": price if price > 0 else None,
                    "market_value": market_value,
                    "weight_pct": (market_value / total_value) if total_value > 0 else 0.0,
                    "unrealized_pnl": unrealized_pnl,
                    "return_pct": return_pct,
                    "opened_at": opened_at.isoformat() if opened_at is not None else None,
                    "days_held": (datetime.now(timezone.utc) - opened_at).days if opened_at is not None else None,
                    "last_reason": holding.last_reason,
                }
            )
        holdings.sort(key=lambda item: float(item.get("market_value") or 0.0), reverse=True)

        realized_map = {str(key).upper(): float(value or 0.0) for key, value in dict(metadata.get("realized_pnl_by_ticker") or {}).items()}
        attribution: list[dict[str, Any]] = []
        seen_tickers = {str(item.get("ticker") or "").upper() for item in holdings}
        for item in holdings:
            ticker = str(item.get("ticker") or "").upper()
            contribution = float(item.get("unrealized_pnl") or 0.0) + realized_map.get(ticker, 0.0)
            attribution.append(
                {
                    "ticker": ticker,
                    "label": item.get("label") or ticker,
                    "asset_type": item.get("asset_type") or "asset",
                    "unrealized_pnl": float(item.get("unrealized_pnl") or 0.0),
                    "realized_pnl": realized_map.get(ticker, 0.0),
                    "total_pnl": contribution,
                }
            )
        for ticker, realized in realized_map.items():
            if ticker in seen_tickers or abs(realized) < 0.005:
                continue
            attribution.append(
                {
                    "ticker": ticker,
                    "label": str(labels_by_ticker.get(ticker) or ticker),
                    "asset_type": self._infer_asset_type(ticker=ticker, label=str(labels_by_ticker.get(ticker) or ticker), source="history"),
                    "unrealized_pnl": 0.0,
                    "realized_pnl": realized,
                    "total_pnl": realized,
                }
            )
        attribution.sort(key=lambda item: float(item.get("total_pnl") or 0.0), reverse=True)

        unrealized_pnl = float(valuation["unrealized_pnl"])
        total_pnl = state.realized_pnl + unrealized_pnl
        return_pct = (total_pnl / state.starting_cash) if state.starting_cash > 0 else 0.0
        cash_pct = (state.cash / total_value) if total_value > 0 else 1.0
        return {
            "portfolio_key": state.portfolio_key,
            "profile_name": str(metadata.get("profile_name") or self.default_profile_name),
            "allowed_asset_types": tuple(str(item).lower() for item in metadata.get("allowed_asset_types") or self.allowed_asset_types),
            "starting_cash": state.starting_cash,
            "cash": state.cash,
            "cash_pct": cash_pct,
            "position_count": len(holdings),
            "holdings": holdings,
            "realized_pnl": state.realized_pnl,
            "unrealized_pnl": unrealized_pnl,
            "total_pnl": total_pnl,
            "total_value": total_value,
            "return_pct": return_pct,
            "last_rebalanced_at": state.last_rebalanced_at.isoformat() if state.last_rebalanced_at is not None else None,
            "last_action_summary": state.last_action_summary,
            "attribution": attribution[:6],
            "top_contributor": attribution[0] if attribution else None,
            "top_detractor": min(attribution, key=lambda item: float(item.get("total_pnl") or 0.0)) if attribution else None,
        }

    async def render_portfolio_text(self, *, conversation_key: str | None, refresh: bool = False) -> str:
        if refresh:
            result = await self.maybe_rebalance(conversation_key=conversation_key, reason="portfolio_view_refresh", force=True)
            return result.rendered_summary
        snapshot = await self.build_snapshot(conversation_key=conversation_key)
        return self.render_snapshot_text(snapshot)

    async def render_trades_text(self, *, conversation_key: str | None, limit: int = 10) -> str:
        state = self.ensure_portfolio(conversation_key)
        trades = self.state_store.list_trades(self.portfolio_key(conversation_key), limit=max(1, limit))
        metadata = self._normalized_metadata(state)
        if not trades:
            return (
                "AI Simulated Portfolio Trades\n"
                f"profile: {metadata.get('profile_name')}\n"
                "ยังไม่มีรายการซื้อขาย"
            )
        lines = [
            "AI Simulated Portfolio Trades",
            f"profile: {metadata.get('profile_name')} | recent={len(trades)}",
        ]
        for trade in trades[:limit]:
            time_text = trade.occurred_at.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            confidence_text = f" | conf {trade.confidence_score:.2f}" if trade.confidence_score is not None else ""
            coverage_text = f" | cov {trade.coverage_score:.2f}" if trade.coverage_score is not None else ""
            rationale_text = f" | {trade.rationale}" if trade.rationale else ""
            lines.append(
                f"- {time_text} | {trade.action.upper()} {trade.ticker} "
                f"qty {trade.quantity:.4f} @ {trade.price:.2f} | ${trade.notional:.2f}{confidence_text}{coverage_text}{rationale_text}"
            )
        return "\n".join(lines)

    async def render_performance_text(self, *, conversation_key: str | None) -> str:
        snapshot = await self.build_snapshot(conversation_key=conversation_key)
        lines = [
            "AI Simulated Portfolio Performance",
            f"profile: {snapshot.get('profile_name')} | return {float(snapshot.get('return_pct') or 0.0):+.2%}",
            (
                f"value ${float(snapshot.get('total_value') or 0.0):,.2f} | "
                f"realized ${float(snapshot.get('realized_pnl') or 0.0):+,.2f} | "
                f"unrealized ${float(snapshot.get('unrealized_pnl') or 0.0):+,.2f}"
            ),
        ]
        attribution = snapshot.get("attribution")
        if isinstance(attribution, list) and attribution:
            lines.append("Attribution")
            for item in attribution[:5]:
                lines.append(
                    f"- {item.get('ticker')}: total ${float(item.get('total_pnl') or 0.0):+,.2f} | "
                    f"unrealized ${float(item.get('unrealized_pnl') or 0.0):+,.2f} | "
                    f"realized ${float(item.get('realized_pnl') or 0.0):+,.2f}"
                )
        return "\n".join(lines)

    async def reset_portfolio(
        self,
        *,
        conversation_key: str | None,
        starting_cash: float | None = None,
        profile_name: str | None = None,
    ) -> AISimulatedPortfolioState:
        previous = self.ensure_portfolio(conversation_key)
        normalized_profile = self._normalize_profile_name(
            profile_name or str(previous.metadata.get("profile_name") or self.default_profile_name)
        )
        state = self.state_store.reset_portfolio(
            self.portfolio_key(conversation_key),
            starting_cash=max(100.0, float(starting_cash if starting_cash is not None else self.starting_cash_usd)),
            base_currency="USD",
        )
        metadata = self._normalized_metadata(state)
        metadata["profile_name"] = normalized_profile
        metadata["allowed_asset_types"] = list(previous.metadata.get("allowed_asset_types") or self.allowed_asset_types)
        metadata["realized_pnl_by_ticker"] = {}
        metadata["last_exit_at_by_ticker"] = {}
        metadata["labels_by_ticker"] = {}
        return self.state_store.save_portfolio(
            self.portfolio_key(conversation_key),
            starting_cash=state.starting_cash,
            cash=state.cash,
            realized_pnl=0.0,
            holdings=[],
            last_rebalanced_at=None,
            last_action_summary="portfolio reset",
            metadata=metadata,
        )

    def status(self, conversation_key: str | None) -> dict[str, Any]:
        state = self.ensure_portfolio(conversation_key)
        metadata = self._normalized_metadata(state)
        return {
            "enabled": self.enabled,
            "backend": self.state_store.backend_name,
            "portfolio_key": state.portfolio_key,
            "holding_count": len(state.holdings),
            "last_rebalanced_at": state.last_rebalanced_at.isoformat() if state.last_rebalanced_at is not None else None,
            "last_action_summary": state.last_action_summary,
            "profile_name": metadata.get("profile_name"),
            "allowed_asset_types": list(metadata.get("allowed_asset_types") or []),
            "rebalance_interval_minutes": self.rebalance_interval_minutes,
        }

    def render_snapshot_text(self, snapshot: Mapping[str, Any]) -> str:
        holdings = snapshot.get("holdings")
        attribution = snapshot.get("attribution")
        allowed_types = ", ".join(str(item) for item in snapshot.get("allowed_asset_types") or ())
        lines = [
            "AI Simulated Portfolio",
            (
                f"profile: {snapshot.get('profile_name')} | universe: {allowed_types or '-'} | "
                f"value ${float(snapshot.get('total_value') or 0.0):,.2f}"
            ),
            (
                f"return {float(snapshot.get('return_pct') or 0.0):+.2%} | "
                f"cash ${float(snapshot.get('cash') or 0.0):,.2f} ({float(snapshot.get('cash_pct') or 0.0):.0%}) | "
                f"positions {int(snapshot.get('position_count') or 0)}"
            ),
            (
                f"realized ${float(snapshot.get('realized_pnl') or 0.0):+,.2f} | "
                f"unrealized ${float(snapshot.get('unrealized_pnl') or 0.0):+,.2f}"
            ),
        ]
        if snapshot.get("last_action_summary"):
            lines.append(f"last action: {snapshot.get('last_action_summary')}")
        if isinstance(holdings, list) and holdings:
            lines.append("Holdings")
            for item in holdings[:6]:
                return_text = f" | ret {float(item.get('return_pct') or 0.0):+.2%}" if item.get("return_pct") is not None else ""
                reason_text = f" | {item.get('last_reason')}" if item.get("last_reason") else ""
                lines.append(
                    f"- {item.get('ticker')}: ${float(item.get('market_value') or 0.0):,.2f} "
                    f"({float(item.get('weight_pct') or 0.0):.0%}) | qty {float(item.get('quantity') or 0.0):.4f}{return_text}{reason_text}"
                )
        else:
            lines.append("Holdings\n- cash only")
        if isinstance(attribution, list) and attribution:
            lines.append("Attribution")
            for item in attribution[:3]:
                lines.append(
                    f"- {item.get('ticker')}: total ${float(item.get('total_pnl') or 0.0):+,.2f} "
                    f"| unrealized ${float(item.get('unrealized_pnl') or 0.0):+,.2f} "
                    f"| realized ${float(item.get('realized_pnl') or 0.0):+,.2f}"
                )
        return "\n".join(lines)

    def render_report_summary_text(self, snapshot: Mapping[str, Any]) -> str:
        holdings = snapshot.get("holdings")
        top_holdings: list[str] = []
        if isinstance(holdings, list):
            for item in holdings[:3]:
                top_holdings.append(f"{item.get('ticker')} {float(item.get('weight_pct') or 0.0):.0%}")
        contributor = snapshot.get("top_contributor") if isinstance(snapshot.get("top_contributor"), Mapping) else None
        detractor = snapshot.get("top_detractor") if isinstance(snapshot.get("top_detractor"), Mapping) else None
        lines = [
            "AI Portfolio",
            (
                f"profile {snapshot.get('profile_name')} | value ${float(snapshot.get('total_value') or 0.0):,.0f} | "
                f"return {float(snapshot.get('return_pct') or 0.0):+.1%} | cash {float(snapshot.get('cash_pct') or 0.0):.0%}"
            ),
            f"holdings: {', '.join(top_holdings) if top_holdings else 'cash only'}",
        ]
        if contributor is not None:
            lines.append(f"best: {contributor.get('ticker')} ${float(contributor.get('total_pnl') or 0.0):+,.0f}")
        if detractor is not None and str(detractor.get('ticker') or '') != str((contributor or {}).get('ticker') or ''):
            lines.append(f"worst: {detractor.get('ticker')} ${float(detractor.get('total_pnl') or 0.0):+,.0f}")
        if snapshot.get("last_action_summary"):
            lines.append(f"action: {snapshot.get('last_action_summary')}")
        return "\n".join(lines)

    def render_rebalance_text(self, result: AISimulatedPortfolioRebalanceResult) -> str:
        return result.rendered_summary

    def _resolve_policy(self, state: AISimulatedPortfolioState) -> AISimulatedPolicy:
        metadata = self._normalized_metadata(state)
        profile_name = self._normalize_profile_name(str(metadata.get("profile_name") or self.default_profile_name))
        base_policy = PROFILE_POLICIES.get(profile_name, PROFILE_POLICIES[self.default_profile_name])
        return AISimulatedPolicy(
            name=base_policy.name,
            max_positions=min(base_policy.max_positions, self.default_max_positions),
            max_position_pct=min(base_policy.max_position_pct, self.default_max_position_pct),
            min_cash_pct=max(base_policy.min_cash_pct, self.default_min_cash_pct),
            min_market_confidence=base_policy.min_market_confidence,
            min_stock_confidence=base_policy.min_stock_confidence,
            min_coverage_score=base_policy.min_coverage_score,
            min_hold_days=base_policy.min_hold_days,
            reentry_cooldown_days=base_policy.reentry_cooldown_days,
            max_turnover_pct=base_policy.max_turnover_pct,
        )

    def _state_to_portfolio_holdings(self, state: AISimulatedPortfolioState) -> tuple[PortfolioHolding, ...]:
        holdings = [
            PortfolioHolding(
                ticker=item.normalized_ticker,
                quantity=float(item.quantity),
                avg_cost=item.avg_cost,
                note=item.label,
            )
            for item in state.holdings
            if item.quantity > 0
        ]
        return tuple(holdings)

    def _select_candidates(
        self,
        payload: Mapping[str, Any],
        *,
        policy: AISimulatedPolicy,
        abstain_mode: bool,
    ) -> list[dict[str, Any]]:
        selected: dict[str, dict[str, Any]] = {}
        allowed_asset_types = {str(item).lower() for item in (self.allowed_asset_types or ())}
        stock_picks = payload.get("stock_picks")
        if isinstance(stock_picks, list):
            for item in stock_picks:
                if not isinstance(item, Mapping):
                    continue
                ticker = str(item.get("ticker") or "").strip().upper()
                if not ticker:
                    continue
                confidence = self._coerce_float(item.get("confidence_score")) or 0.0
                coverage = self._coerce_float(item.get("coverage_score")) or 0.0
                if confidence < policy.min_stock_confidence or coverage < policy.min_coverage_score:
                    continue
                asset_type = self._infer_asset_type(
                    ticker=ticker,
                    label=str(item.get("company_name") or item.get("label") or ticker),
                    source="stock_pick",
                )
                if asset_type not in allowed_asset_types:
                    continue
                if abstain_mode and asset_type == "stock":
                    continue
                stance = str(item.get("stance") or "").strip().lower()
                if stance and stance not in {"buy", "accumulate", "add", "hold", "overweight"}:
                    continue
                selected[ticker] = {
                    "ticker": ticker,
                    "label": str(item.get("company_name") or item.get("label") or ticker),
                    "asset_type": asset_type,
                    "confidence_score": confidence,
                    "coverage_score": coverage,
                    "score": self._coerce_float(item.get("score")) or (confidence * 10.0),
                    "source": "stock_picks",
                    "reason": f"top pick confidence {confidence:.2f}, coverage {coverage:.2f}",
                }
        asset_snapshots = payload.get("asset_snapshots")
        if isinstance(asset_snapshots, list):
            for item in asset_snapshots:
                if not isinstance(item, Mapping):
                    continue
                ticker = str(item.get("ticker") or "").strip().upper()
                if not ticker:
                    continue
                label = str(item.get("label") or item.get("company_name") or ticker)
                asset_type = self._infer_asset_type(ticker=ticker, label=label, source="asset_snapshot")
                if asset_type not in allowed_asset_types:
                    continue
                if ticker not in self.core_tickers and asset_type == "etf" and not abstain_mode:
                    continue
                if abstain_mode and asset_type not in {"etf", "gold"}:
                    continue
                coverage = self._coerce_float(item.get("coverage_score")) or 0.0
                if coverage < max(0.35, policy.min_coverage_score - 0.08):
                    continue
                trend = str(item.get("trend") or "").strip().lower()
                trend_score = self._coerce_float(item.get("trend_score")) or 0.0
                if trend and trend in {"downtrend", "weak"} and ticker not in RISK_OFF_SAFE_TICKERS:
                    continue
                selected.setdefault(
                    ticker,
                    {
                        "ticker": ticker,
                        "label": label,
                        "asset_type": asset_type,
                        "confidence_score": max(0.45, min(0.90, 0.45 + trend_score / 10.0)),
                        "coverage_score": coverage,
                        "score": trend_score or coverage * 10.0,
                        "source": "asset_snapshots",
                        "reason": f"{label} trend {trend or 'mixed'} | coverage {coverage:.2f}",
                    },
                )
        ranked = sorted(
            selected.values(),
            key=lambda item: (
                float(item.get("score") or 0.0),
                float(item.get("confidence_score") or 0.0),
                float(item.get("coverage_score") or 0.0),
            ),
            reverse=True,
        )
        return ranked[: max(policy.max_positions * 2, 6)]

    def _build_target_weights(
        self,
        *,
        candidates: Sequence[Mapping[str, Any]],
        policy: AISimulatedPolicy,
        abstain_mode: bool,
        deep_risk_off: bool,
    ) -> dict[str, float]:
        if deep_risk_off:
            selected = [item for item in candidates if str(item.get("ticker") or "").upper() in RISK_OFF_SAFE_TICKERS]
        elif abstain_mode:
            selected = [item for item in candidates if str(item.get("asset_type") or "") in {"etf", "gold"}]
        else:
            selected = list(candidates)
        if not selected:
            return {}
        selected = selected[: policy.max_positions]
        investable_pct = max(0.0, 1.0 - policy.min_cash_pct)
        if deep_risk_off:
            investable_pct = min(investable_pct, 0.45)
        elif abstain_mode:
            investable_pct = min(investable_pct, 0.65)
        raw_scores: list[tuple[str, float]] = []
        for item in selected:
            confidence = float(item.get("confidence_score") or 0.0)
            coverage = float(item.get("coverage_score") or 0.0)
            score = float(item.get("score") or 0.0)
            raw = max(0.01, confidence * 0.55 + coverage * 0.35 + min(score / 10.0, 1.0) * 0.10)
            raw_scores.append((str(item.get("ticker") or "").upper(), raw))
        total_raw = sum(score for _, score in raw_scores) or 1.0
        weights = {ticker: min(policy.max_position_pct, investable_pct * (score / total_raw)) for ticker, score in raw_scores}
        allocated = sum(weights.values())
        if allocated > investable_pct and allocated > 0:
            scale = investable_pct / allocated
            weights = {ticker: value * scale for ticker, value in weights.items()}
        return {ticker: round(value, 6) for ticker, value in weights.items() if value > 0}

    def _compute_valuation(
        self,
        *,
        state: AISimulatedPortfolioState,
        quotes: Mapping[str, AssetQuote | None],
    ) -> dict[str, float]:
        holdings_value = 0.0
        unrealized_pnl = 0.0
        for holding in state.holdings:
            quote = quotes.get(holding.normalized_ticker)
            price = self._coerce_float(quote.price if quote is not None else None)
            if price is None or price <= 0:
                price = self._coerce_float(holding.avg_cost) or 0.0
            market_value = holding.quantity * price
            holdings_value += market_value
            if holding.avg_cost not in (None, 0.0) and price > 0:
                unrealized_pnl += (price - float(holding.avg_cost)) * holding.quantity
        total_value = state.cash + holdings_value
        return {
            "holdings_value": holdings_value,
            "cash": state.cash,
            "total_value": total_value,
            "unrealized_pnl": unrealized_pnl,
        }

    async def _execute_rebalance(
        self,
        *,
        conversation_key: str | None,
        reason: str,
        state: AISimulatedPortfolioState,
        payload: Mapping[str, Any],
        policy: AISimulatedPolicy,
        abstain_mode: bool,
        deep_risk_off: bool,
        candidates: Sequence[Mapping[str, Any]],
        quotes: Mapping[str, AssetQuote | None],
        target_value_by_ticker: Mapping[str, float],
        total_value: float,
        cash_floor_value: float,
    ) -> AISimulatedPortfolioRebalanceResult:
        now = datetime.now(timezone.utc)
        metadata = self._normalized_metadata(state)
        labels_by_ticker = dict(metadata.get("labels_by_ticker") or {})
        realized_by_ticker = {
            str(key).upper(): float(value or 0.0)
            for key, value in dict(metadata.get("realized_pnl_by_ticker") or {}).items()
        }
        last_exit_at_by_ticker = {
            str(key).upper(): str(value or "")
            for key, value in dict(metadata.get("last_exit_at_by_ticker") or {}).items()
        }
        holdings = {item.normalized_ticker: item for item in state.holdings}
        decisions: list[dict[str, Any]] = []
        trade_payloads: list[dict[str, Any]] = []
        action_count = 0

        turnover_cap = total_value * (1.0 if deep_risk_off else policy.max_turnover_pct)
        turnover_used = 0.0

        def sell_priority(item: AISimulatedPortfolioHolding) -> tuple[int, float]:
            target_value = float(target_value_by_ticker.get(item.normalized_ticker) or 0.0)
            quote = quotes.get(item.normalized_ticker)
            price = self._coerce_float(quote.price if quote is not None else None) or self._coerce_float(item.avg_cost) or 0.0
            current_value = item.quantity * price
            return (0 if target_value <= 0 else 1, current_value)

        for holding in sorted(holdings.values(), key=sell_priority):
            ticker = holding.normalized_ticker
            quote = quotes.get(ticker)
            price = self._coerce_float(quote.price if quote is not None else None) or self._coerce_float(holding.avg_cost)
            if price is None or price <= 0:
                continue
            current_value = holding.quantity * price
            target_value = float(target_value_by_ticker.get(ticker) or 0.0)
            reduce_value = max(0.0, current_value - target_value)
            if reduce_value < self.min_trade_notional_usd:
                continue
            can_force_sell = deep_risk_off and ticker not in RISK_OFF_SAFE_TICKERS
            if not can_force_sell and not self._can_sell_holding(holding=holding, policy=policy, now=now):
                decisions.append({"action": "hold", "ticker": ticker, "reason": "min_hold_period"})
                continue
            remaining_turnover = max(0.0, turnover_cap - turnover_used)
            if remaining_turnover < self.min_trade_notional_usd:
                decisions.append({"action": "hold", "ticker": ticker, "reason": "turnover_cap"})
                continue
            trade_value = min(reduce_value, remaining_turnover)
            quantity = trade_value / price
            if not self.allow_fractional:
                quantity = float(int(quantity))
            if quantity <= 0:
                continue
            if quantity >= holding.quantity * 0.98:
                quantity = holding.quantity
                trade_value = quantity * price
            elif quantity * price < self.min_trade_notional_usd:
                continue
            remaining_quantity = max(0.0, holding.quantity - quantity)
            cash = state.cash + trade_value
            if remaining_quantity <= 0:
                holdings.pop(ticker, None)
                last_exit_at_by_ticker[ticker] = now.isoformat()
            else:
                holdings[ticker] = AISimulatedPortfolioHolding(
                    ticker=ticker,
                    quantity=remaining_quantity,
                    avg_cost=holding.avg_cost,
                    label=holding.label,
                    asset_type=holding.asset_type,
                    last_reason=holding.last_reason,
                    opened_at=holding.opened_at,
                )
            realized_increment = 0.0
            if holding.avg_cost not in (None, 0.0):
                realized_increment = (price - float(holding.avg_cost)) * quantity
                realized_by_ticker[ticker] = realized_by_ticker.get(ticker, 0.0) + realized_increment
            state = AISimulatedPortfolioState(
                portfolio_key=state.portfolio_key,
                starting_cash=state.starting_cash,
                cash=cash,
                realized_pnl=state.realized_pnl + realized_increment,
                created_at=state.created_at,
                updated_at=state.updated_at,
                last_rebalanced_at=state.last_rebalanced_at,
                last_action_summary=state.last_action_summary,
                holdings=tuple(holdings.values()),
                metadata=state.metadata,
            )
            turnover_used += trade_value
            trade_payloads.append(
                self._build_trade_payload(
                    action="sell",
                    ticker=ticker,
                    quantity=quantity,
                    price=price,
                    label=holding.label or ticker,
                    asset_type=holding.asset_type,
                    rationale="risk reduction" if can_force_sell else "rebalance trim",
                    confidence_score=None,
                    coverage_score=None,
                    occurred_at=now,
                    detail={"reason": reason, "policy": policy.name},
                )
            )
            decisions.append({"action": "sell", "ticker": ticker, "notional": trade_value, "reason": "rebalance"})
            action_count += 1

        available_cash = max(0.0, state.cash - cash_floor_value)
        for ticker, target_value in sorted(target_value_by_ticker.items(), key=lambda item: item[1], reverse=True):
            candidate = self._lookup_candidate(candidates, ticker)
            if candidate is None:
                continue
            quote = quotes.get(ticker)
            price = self._coerce_float(quote.price if quote is not None else None)
            if price is None or price <= 0:
                decisions.append({"action": "skip", "ticker": ticker, "reason": "missing_quote"})
                continue
            existing = holdings.get(ticker)
            current_value = (existing.quantity * price) if existing is not None else 0.0
            needed_value = max(0.0, target_value - current_value)
            if needed_value < self.min_trade_notional_usd:
                continue
            remaining_turnover = max(0.0, turnover_cap - turnover_used)
            buy_budget = min(needed_value, available_cash, remaining_turnover)
            if buy_budget < self.min_trade_notional_usd:
                decisions.append({"action": "hold", "ticker": ticker, "reason": "cash_or_turnover_limit"})
                continue
            if self._in_reentry_cooldown(
                ticker=ticker,
                last_exit_at_by_ticker=last_exit_at_by_ticker,
                policy=policy,
                now=now,
            ):
                decisions.append({"action": "hold", "ticker": ticker, "reason": "reentry_cooldown"})
                continue
            quantity = buy_budget / price
            if not self.allow_fractional:
                quantity = float(int(quantity))
            if quantity <= 0 or quantity * price < self.min_trade_notional_usd:
                continue
            actual_notional = quantity * price
            if actual_notional > available_cash + 1e-6:
                continue
            if existing is None:
                holdings[ticker] = AISimulatedPortfolioHolding(
                    ticker=ticker,
                    quantity=quantity,
                    avg_cost=price,
                    label=str(candidate.get("label") or ticker),
                    asset_type=str(candidate.get("asset_type") or "asset"),
                    last_reason=str(candidate.get("reason") or "new position"),
                    opened_at=now,
                )
            else:
                total_qty = existing.quantity + quantity
                avg_cost = (((existing.quantity * float(existing.avg_cost or price)) + actual_notional) / total_qty) if total_qty > 0 else price
                holdings[ticker] = AISimulatedPortfolioHolding(
                    ticker=ticker,
                    quantity=total_qty,
                    avg_cost=avg_cost,
                    label=existing.label or str(candidate.get("label") or ticker),
                    asset_type=existing.asset_type or str(candidate.get("asset_type") or "asset"),
                    last_reason=str(candidate.get("reason") or existing.last_reason or "add"),
                    opened_at=existing.opened_at or now,
                )
            state = AISimulatedPortfolioState(
                portfolio_key=state.portfolio_key,
                starting_cash=state.starting_cash,
                cash=state.cash - actual_notional,
                realized_pnl=state.realized_pnl,
                created_at=state.created_at,
                updated_at=state.updated_at,
                last_rebalanced_at=state.last_rebalanced_at,
                last_action_summary=state.last_action_summary,
                holdings=tuple(holdings.values()),
                metadata=state.metadata,
            )
            labels_by_ticker[ticker] = str(candidate.get("label") or ticker)
            turnover_used += actual_notional
            available_cash = max(0.0, state.cash - cash_floor_value)
            trade_payloads.append(
                self._build_trade_payload(
                    action="buy",
                    ticker=ticker,
                    quantity=quantity,
                    price=price,
                    label=str(candidate.get("label") or ticker),
                    asset_type=str(candidate.get("asset_type") or "asset"),
                    rationale=str(candidate.get("reason") or "rebalance add"),
                    confidence_score=self._coerce_float(candidate.get("confidence_score")),
                    coverage_score=self._coerce_float(candidate.get("coverage_score")),
                    occurred_at=now,
                    detail={"reason": reason, "policy": policy.name, "market_confidence": self._market_confidence_score(payload)},
                )
            )
            decisions.append({"action": "buy", "ticker": ticker, "notional": actual_notional, "reason": "rebalance"})
            action_count += 1

        metadata["realized_pnl_by_ticker"] = realized_by_ticker
        metadata["last_exit_at_by_ticker"] = last_exit_at_by_ticker
        metadata["labels_by_ticker"] = labels_by_ticker
        action_summary = self._build_action_summary(
            decisions=decisions,
            abstain_mode=abstain_mode,
            deep_risk_off=deep_risk_off,
            market_confidence=self._market_confidence_score(payload),
            policy=policy,
        )
        saved_state = self.state_store.save_portfolio(
            self.portfolio_key(conversation_key),
            starting_cash=state.starting_cash,
            cash=state.cash,
            realized_pnl=state.realized_pnl,
            holdings=[self._serialize_holding_payload(item) for item in holdings.values() if item.quantity > 0],
            last_rebalanced_at=now,
            last_action_summary=action_summary,
            metadata=metadata,
        )
        saved_trades = self.state_store.append_trades(self.portfolio_key(conversation_key), trade_payloads) if trade_payloads else ()
        snapshot = await self.build_snapshot(conversation_key=conversation_key)
        summary = self.render_snapshot_text(snapshot)
        log_event(
            "ai_simulated_portfolio_rebalance",
            policy=policy.name,
            action_count=action_count,
            abstain_mode=abstain_mode,
            deep_risk_off=deep_risk_off,
            market_confidence=self._market_confidence_score(payload),
        )
        return AISimulatedPortfolioRebalanceResult(
            state=saved_state,
            snapshot=snapshot,
            decisions=tuple(decisions),
            trades=saved_trades,
            action_count=action_count,
            rendered_summary=summary,
            market_payload=dict(payload),
        )

    def _serialize_holding_payload(self, item: AISimulatedPortfolioHolding) -> dict[str, Any]:
        return {
            "ticker": item.normalized_ticker,
            "quantity": float(item.quantity),
            "avg_cost": item.avg_cost,
            "label": item.label,
            "asset_type": item.asset_type,
            "last_reason": item.last_reason,
            "opened_at": item.opened_at.isoformat() if item.opened_at is not None else None,
        }

    def _can_sell_holding(
        self,
        *,
        holding: AISimulatedPortfolioHolding,
        policy: AISimulatedPolicy,
        now: datetime,
    ) -> bool:
        opened_at = holding.opened_at
        if opened_at is None:
            return True
        return (now - opened_at) >= timedelta(days=policy.min_hold_days)

    def _in_reentry_cooldown(
        self,
        *,
        ticker: str,
        last_exit_at_by_ticker: Mapping[str, Any],
        policy: AISimulatedPolicy,
        now: datetime,
    ) -> bool:
        last_exit_at = self._parse_datetime(last_exit_at_by_ticker.get(ticker))
        if last_exit_at is None:
            return False
        return (now - last_exit_at) < timedelta(days=policy.reentry_cooldown_days)

    def _build_trade_payload(
        self,
        *,
        action: str,
        ticker: str,
        quantity: float,
        price: float,
        label: str,
        asset_type: str,
        rationale: str,
        confidence_score: float | None,
        coverage_score: float | None,
        occurred_at: datetime,
        detail: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        trade_id = hashlib.sha256(
            f"{ticker}:{action}:{quantity:.8f}:{price:.8f}:{occurred_at.isoformat()}".encode("utf-8")
        ).hexdigest()[:24]
        return {
            "trade_id": trade_id,
            "action": action,
            "ticker": ticker,
            "quantity": round(quantity, 8),
            "price": round(price, 6),
            "notional": round(quantity * price, 6),
            "occurred_at": occurred_at.isoformat(),
            "label": label,
            "asset_type": asset_type,
            "rationale": rationale,
            "confidence_score": confidence_score,
            "coverage_score": coverage_score,
            "detail": dict(detail or {}),
        }

    def _build_action_summary(
        self,
        *,
        decisions: Sequence[Mapping[str, Any]],
        abstain_mode: bool,
        deep_risk_off: bool,
        market_confidence: float,
        policy: AISimulatedPolicy,
    ) -> str:
        buy_count = sum(1 for item in decisions if str(item.get("action") or "") == "buy")
        sell_count = sum(1 for item in decisions if str(item.get("action") or "") == "sell")
        if deep_risk_off:
            posture = "deep risk-off"
        elif abstain_mode:
            posture = "abstain / defensive"
        else:
            posture = "risk-on"
        if buy_count == 0 and sell_count == 0:
            return f"no trade | posture={posture} | market_confidence={market_confidence:.2f} | profile={policy.name}"
        return (
            f"buys={buy_count} sells={sell_count} | posture={posture} | "
            f"market_confidence={market_confidence:.2f} | profile={policy.name}"
        )

    def _lookup_candidate(self, candidates: Sequence[Mapping[str, Any]], ticker: str) -> Mapping[str, Any] | None:
        normalized = ticker.strip().upper()
        for item in candidates:
            if str(item.get("ticker") or "").strip().upper() == normalized:
                return item
        return None

    def _market_confidence_score(self, payload: Mapping[str, Any]) -> float:
        market_confidence = payload.get("market_confidence")
        if isinstance(market_confidence, Mapping):
            value = self._coerce_float(market_confidence.get("score"))
            if value is not None:
                return max(0.0, min(1.0, value))
        return 0.50

    def _coerce_float(self, value: Any) -> float | None:
        try:
            if value is None:
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    def _normalize_profile_name(self, value: str | None) -> str:
        return str(normalize_profile_name(value or "growth", default="growth"))

    def _parse_datetime(self, value: Any) -> datetime | None:
        if isinstance(value, datetime):
            return value.astimezone(timezone.utc) if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
        if isinstance(value, str) and value.strip():
            try:
                parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
            except ValueError:
                return None
            return parsed.astimezone(timezone.utc) if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)
        return None

    def _normalized_metadata(self, state: AISimulatedPortfolioState) -> dict[str, Any]:
        metadata = dict(state.metadata)
        metadata.setdefault("engine", "ai-simulated-portfolio")
        metadata["profile_name"] = self._normalize_profile_name(str(metadata.get("profile_name") or self.default_profile_name))
        allowed_asset_types = metadata.get("allowed_asset_types") or self.allowed_asset_types
        normalized_asset_types = tuple(
            dict.fromkeys(str(item).strip().lower() for item in allowed_asset_types if str(item).strip())
        )
        metadata["allowed_asset_types"] = list(normalized_asset_types or self.allowed_asset_types)
        metadata["realized_pnl_by_ticker"] = dict(metadata.get("realized_pnl_by_ticker") or {})
        metadata["last_exit_at_by_ticker"] = dict(metadata.get("last_exit_at_by_ticker") or {})
        metadata["labels_by_ticker"] = dict(metadata.get("labels_by_ticker") or {})
        return metadata

    def _infer_asset_type(self, *, ticker: str, label: str, source: str) -> str:
        normalized_ticker = ticker.strip().upper()
        normalized_label = label.strip().lower()
        if normalized_ticker in {"GLD", "IAU", "GOLD"} or "gold" in normalized_label:
            return "gold"
        if normalized_ticker in self.core_tickers or "etf" in normalized_label or source == "asset_snapshot":
            return "etf"
        return "stock"
