from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from threading import RLock
from typing import Any, Mapping, Sequence

import httpx
from loguru import logger

from invest_advisor_bot.runtime_diagnostics import diagnostics


@dataclass(slots=True, frozen=True)
class BrokerAccountSnapshot:
    account_id: str | None
    status: str | None
    currency: str | None
    equity: float | None
    buying_power: float | None
    cash: float | None
    pattern_day_trader: bool | None
    updated_at: datetime


@dataclass(slots=True, frozen=True)
class BrokerPosition:
    symbol: str
    qty: float
    side: str | None
    market_value: float | None
    cost_basis: float | None
    unrealized_pl: float | None
    unrealized_plpc: float | None


@dataclass(slots=True, frozen=True)
class BrokerOrderResult:
    order_id: str | None
    client_order_id: str | None
    symbol: str
    side: str
    type: str
    status: str | None
    qty: float
    limit_price: float | None
    submitted_at: datetime | None
    raw: dict[str, Any]


class ExecutionSandboxClient:
    """Optional broker paper-trading client using Alpaca or Tradier sandbox APIs."""

    def __init__(
        self,
        *,
        provider: str = "alpaca",
        enabled: bool = False,
        api_key: str = "",
        api_secret: str = "",
        base_url: str = "https://paper-api.alpaca.markets",
        tradier_access_token: str = "",
        tradier_account_id: str = "",
        tradier_base_url: str = "https://sandbox.tradier.com",
        timeout_seconds: float = 12.0,
    ) -> None:
        self.provider = str(provider or "alpaca").strip().casefold() or "alpaca"
        self.enabled_flag = bool(enabled)
        self.api_key = api_key.strip()
        self.api_secret = api_secret.strip()
        self.base_url = base_url.strip().rstrip("/") or "https://paper-api.alpaca.markets"
        self.tradier_access_token = tradier_access_token.strip()
        self.tradier_account_id = tradier_account_id.strip()
        self.tradier_base_url = tradier_base_url.strip().rstrip("/") or "https://sandbox.tradier.com"
        self.timeout_seconds = max(2.0, float(timeout_seconds))
        self._http_client: httpx.Client | None = None
        self._lock = RLock()
        self._warning: str | None = None
        self._last_order_id: str | None = None
        self._last_order_symbol: str | None = None
        self._last_order_kind: str | None = None

    def enabled(self) -> bool:
        if not self.enabled_flag:
            return False
        if self.provider == "alpaca":
            return bool(self.api_key and self.api_secret)
        if self.provider == "tradier":
            return bool(self.tradier_access_token and self.tradier_account_id)
        return False

    def status(self) -> dict[str, Any]:
        return {
            "available": self.enabled(),
            "provider": self.provider,
            "enabled": self.enabled_flag,
            "configured": bool(self.api_key and self.api_secret) if self.provider == "alpaca" else bool(self.tradier_access_token and self.tradier_account_id),
            "base_url": self.base_url if self.provider == "alpaca" else self.tradier_base_url,
            "tradier_account_id": self.tradier_account_id or None,
            "warning": self._warning,
            "last_order_id": self._last_order_id,
            "last_order_symbol": self._last_order_symbol,
            "last_order_kind": self._last_order_kind,
        }

    async def get_account(self) -> BrokerAccountSnapshot | None:
        return await asyncio.to_thread(self._get_account_sync)

    async def list_positions(self) -> list[BrokerPosition]:
        return await asyncio.to_thread(self._list_positions_sync)

    async def submit_order(
        self,
        *,
        symbol: str,
        qty: float,
        side: str,
        order_type: str = "market",
        time_in_force: str = "day",
        limit_price: float | None = None,
    ) -> BrokerOrderResult | None:
        return await asyncio.to_thread(
            self._submit_order_sync,
            symbol,
            qty,
            side,
            order_type,
            time_in_force,
            limit_price,
        )

    async def submit_option_order(
        self,
        *,
        contract_symbol: str,
        qty: float,
        side: str,
        order_type: str = "market",
        time_in_force: str = "day",
        limit_price: float | None = None,
    ) -> BrokerOrderResult | None:
        return await asyncio.to_thread(
            self._submit_option_order_sync,
            contract_symbol,
            qty,
            side,
            order_type,
            time_in_force,
            limit_price,
        )

    def _get_account_sync(self) -> BrokerAccountSnapshot | None:
        if not self.enabled():
            self._warning = "broker sandbox disabled or missing API credentials"
            return None
        started_at = self._monotonic()
        try:
            payload = self._request("GET", "account")
        except Exception as exc:
            diagnostics.record_provider_latency(
                service="broker_client",
                provider=self.provider,
                operation="account",
                latency_ms=max(0.0, (self._monotonic() - started_at) * 1000.0),
                success=False,
            )
            logger.warning("Broker account request failed: {}", exc)
            self._warning = str(exc)
            return None
        diagnostics.record_provider_latency(
            service="broker_client",
            provider=self.provider,
            operation="account",
            latency_ms=max(0.0, (self._monotonic() - started_at) * 1000.0),
            success=True,
        )
        if not isinstance(payload, Mapping):
            return None
        if self.provider == "tradier":
            balances = payload.get("balances") if isinstance(payload.get("balances"), Mapping) else payload
            return BrokerAccountSnapshot(
                account_id=self.tradier_account_id or self._as_optional_str(payload.get("account_number")),
                status="active" if balances else None,
                currency="USD",
                equity=self._as_float(balances.get("total_equity")) if isinstance(balances, Mapping) else None,
                buying_power=self._as_float(balances.get("stock_buying_power")) if isinstance(balances, Mapping) else None,
                cash=self._as_float(balances.get("cash")) if isinstance(balances, Mapping) else None,
                pattern_day_trader=None,
                updated_at=datetime.now(timezone.utc),
            )
        return BrokerAccountSnapshot(
            account_id=self._as_optional_str(payload.get("id")),
            status=self._as_optional_str(payload.get("status")),
            currency=self._as_optional_str(payload.get("currency")),
            equity=self._as_float(payload.get("equity")),
            buying_power=self._as_float(payload.get("buying_power")),
            cash=self._as_float(payload.get("cash")),
            pattern_day_trader=self._as_bool(payload.get("pattern_day_trader")),
            updated_at=datetime.now(timezone.utc),
        )

    def _list_positions_sync(self) -> list[BrokerPosition]:
        if not self.enabled():
            self._warning = "broker sandbox disabled or missing API credentials"
            return []
        started_at = self._monotonic()
        try:
            payload = self._request("GET", "positions")
        except Exception as exc:
            diagnostics.record_provider_latency(
                service="broker_client",
                provider=self.provider,
                operation="positions",
                latency_ms=(self._monotonic() - started_at) * 1000.0,
                success=False,
            )
            logger.warning("Broker positions request failed: {}", exc)
            self._warning = str(exc)
            return []
        diagnostics.record_provider_latency(
            service="broker_client",
            provider=self.provider,
            operation="positions",
            latency_ms=(self._monotonic() - started_at) * 1000.0,
            success=True,
        )
        if self.provider == "tradier" and isinstance(payload, Mapping):
            positions_root = payload.get("positions") if isinstance(payload.get("positions"), Mapping) else payload
            items = positions_root.get("position") if isinstance(positions_root, Mapping) else []
            if isinstance(items, Mapping):
                payload = [items]
            elif isinstance(items, list):
                payload = items
            else:
                payload = []
        if not isinstance(payload, list):
            return []
        positions: list[BrokerPosition] = []
        for item in payload:
            if not isinstance(item, Mapping):
                continue
            symbol = self._as_optional_str(item.get("symbol"))
            qty = self._as_float(item.get("qty"))
            if not symbol or qty is None:
                continue
            positions.append(
                BrokerPosition(
                    symbol=symbol,
                    qty=qty,
                    side=self._as_optional_str(item.get("side") or item.get("cost_basis")),
                    market_value=self._as_float(item.get("market_value")),
                    cost_basis=self._as_float(item.get("cost_basis")),
                    unrealized_pl=self._as_float(item.get("unrealized_pl")),
                    unrealized_plpc=self._as_float(item.get("unrealized_plpc")),
                )
            )
        return positions

    def _submit_order_sync(
        self,
        symbol: str,
        qty: float,
        side: str,
        order_type: str,
        time_in_force: str,
        limit_price: float | None,
    ) -> BrokerOrderResult | None:
        if not self.enabled():
            self._warning = "broker sandbox disabled or missing API credentials"
            return None
        normalized_symbol = str(symbol or "").strip().upper()
        normalized_side = str(side or "").strip().lower()
        normalized_type = str(order_type or "market").strip().lower()
        if not normalized_symbol or normalized_side not in {"buy", "sell"}:
            return None
        payload = self._build_order_payload(
            symbol=normalized_symbol,
            qty=qty,
            side=normalized_side,
            order_type=normalized_type,
            time_in_force=time_in_force,
            limit_price=limit_price,
            option_symbol=None,
        )
        started_at = self._monotonic()
        try:
            response_payload = self._request("POST", "orders", json_payload=payload)
        except Exception as exc:
            diagnostics.record_provider_latency(
                service="broker_client",
                provider=self.provider,
                operation="submit_order",
                latency_ms=(self._monotonic() - started_at) * 1000.0,
                success=False,
            )
            logger.warning("Broker order submission failed: {}", exc)
            self._warning = str(exc)
            return None
        diagnostics.record_provider_latency(
            service="broker_client",
            provider=self.provider,
            operation="submit_order",
            latency_ms=(self._monotonic() - started_at) * 1000.0,
            success=True,
        )
        if not isinstance(response_payload, Mapping):
            return None
        order_root = response_payload.get("order") if isinstance(response_payload.get("order"), Mapping) else None
        order_id = self._as_optional_str(response_payload.get("id") or (order_root.get("id") if isinstance(order_root, Mapping) else None))
        self._last_order_id = order_id
        self._last_order_symbol = normalized_symbol
        self._last_order_kind = "equity"
        return BrokerOrderResult(
            order_id=order_id,
            client_order_id=self._as_optional_str(response_payload.get("client_order_id")),
            symbol=normalized_symbol,
            side=normalized_side,
            type=normalized_type,
            status=self._as_optional_str(response_payload.get("status") or (order_root.get("status") if isinstance(order_root, Mapping) else None)),
            qty=float(qty),
            limit_price=float(limit_price) if limit_price is not None else None,
            submitted_at=self._parse_datetime(response_payload.get("submitted_at") or (order_root.get("create_date") if isinstance(order_root, Mapping) else None)),
            raw=dict(response_payload),
        )

    def _submit_option_order_sync(
        self,
        contract_symbol: str,
        qty: float,
        side: str,
        order_type: str,
        time_in_force: str,
        limit_price: float | None,
    ) -> BrokerOrderResult | None:
        if self.provider != "tradier" or not self.enabled():
            self._warning = "tradier option execution unavailable"
            return None
        normalized_symbol = str(contract_symbol or "").strip().upper()
        normalized_side = str(side or "").strip().lower()
        normalized_type = str(order_type or "market").strip().lower()
        if not normalized_symbol or normalized_side not in {"buy_to_open", "buy_to_close", "sell_to_open", "sell_to_close"}:
            return None
        payload = self._build_order_payload(
            symbol=normalized_symbol,
            qty=qty,
            side=normalized_side,
            order_type=normalized_type,
            time_in_force=time_in_force,
            limit_price=limit_price,
            option_symbol=normalized_symbol,
        )
        started_at = self._monotonic()
        try:
            response_payload = self._request("POST", "orders", json_payload=payload)
        except Exception as exc:
            diagnostics.record_provider_latency(
                service="broker_client",
                provider=self.provider,
                operation="submit_option_order",
                latency_ms=(self._monotonic() - started_at) * 1000.0,
                success=False,
            )
            logger.warning("Broker option order submission failed: {}", exc)
            self._warning = str(exc)
            return None
        diagnostics.record_provider_latency(
            service="broker_client",
            provider=self.provider,
            operation="submit_option_order",
            latency_ms=(self._monotonic() - started_at) * 1000.0,
            success=True,
        )
        order_root = response_payload.get("order") if isinstance(response_payload, Mapping) and isinstance(response_payload.get("order"), Mapping) else response_payload
        if not isinstance(order_root, Mapping):
            return None
        self._last_order_id = self._as_optional_str(order_root.get("id"))
        self._last_order_symbol = normalized_symbol
        self._last_order_kind = "option"
        return BrokerOrderResult(
            order_id=self._as_optional_str(order_root.get("id")),
            client_order_id=None,
            symbol=normalized_symbol,
            side=normalized_side,
            type=normalized_type,
            status=self._as_optional_str(order_root.get("status")),
            qty=float(qty),
            limit_price=float(limit_price) if limit_price is not None else None,
            submitted_at=self._parse_datetime(order_root.get("create_date")),
            raw=dict(response_payload),
        )

    def _build_order_payload(
        self,
        *,
        symbol: str,
        qty: float,
        side: str,
        order_type: str,
        time_in_force: str,
        limit_price: float | None,
        option_symbol: str | None,
    ) -> dict[str, Any]:
        if self.provider == "tradier":
            base_symbol = symbol
            if option_symbol:
                base_symbol = self._extract_underlying_from_option_symbol(option_symbol)
            payload: dict[str, Any] = {
                "class": "option" if option_symbol else "equity",
                "symbol": base_symbol,
                "duration": str(time_in_force or "day").strip().lower() or "day",
                "side": side,
                "quantity": float(qty),
                "type": order_type,
            }
            if option_symbol:
                payload["option_symbol"] = option_symbol
            if order_type == "limit" and limit_price is not None:
                payload["price"] = float(limit_price)
            return payload
        payload = {
            "symbol": symbol,
            "qty": float(qty),
            "side": side,
            "type": order_type,
            "time_in_force": str(time_in_force or "day").strip().lower() or "day",
        }
        if order_type == "limit" and limit_price is not None:
            payload["limit_price"] = float(limit_price)
        return payload

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, Any] | None = None,
        json_payload: Mapping[str, Any] | None = None,
    ) -> Any:
        if self.provider == "tradier":
            route_map = {
                "account": f"/v1/accounts/{self.tradier_account_id}/balances",
                "positions": f"/v1/accounts/{self.tradier_account_id}/positions",
                "orders": f"/v1/accounts/{self.tradier_account_id}/orders",
            }
            endpoint = route_map.get(path, path)
            response = self._get_http_client().request(
                method=method.upper(),
                url=f"{self.tradier_base_url}{endpoint}",
                headers={
                    "Authorization": f"Bearer {self.tradier_access_token}",
                    "Accept": "application/json",
                    "User-Agent": "invest-advisor-bot/0.2",
                },
                params=params,
                data=json_payload,
            )
        else:
            route_map = {
                "account": "/v2/account",
                "positions": "/v2/positions",
                "orders": "/v2/orders",
            }
            endpoint = route_map.get(path, path)
            response = self._get_http_client().request(
                method=method.upper(),
                url=f"{self.base_url}{endpoint}",
                headers={
                    "APCA-API-KEY-ID": self.api_key,
                    "APCA-API-SECRET-KEY": self.api_secret,
                    "User-Agent": "invest-advisor-bot/0.2",
                },
                params=params,
                json=json_payload,
            )
        response.raise_for_status()
        return response.json()

    def _get_http_client(self) -> httpx.Client:
        with self._lock:
            if self._http_client is None:
                self._http_client = httpx.Client(timeout=self.timeout_seconds)
            return self._http_client

    @staticmethod
    def _parse_datetime(value: Any) -> datetime | None:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
        return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)

    @staticmethod
    def _as_float(value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _as_optional_str(value: Any) -> str | None:
        text = str(value or "").strip()
        return text or None

    @staticmethod
    def _as_bool(value: Any) -> bool | None:
        if value is None:
            return None
        if isinstance(value, bool):
            return value
        text = str(value).strip().casefold()
        if text in {"true", "1", "yes"}:
            return True
        if text in {"false", "0", "no"}:
            return False
        return None

    @staticmethod
    def _extract_underlying_from_option_symbol(symbol: str) -> str:
        text = str(symbol or "").strip().upper()
        letters: list[str] = []
        for char in text:
            if char.isalpha():
                letters.append(char)
                continue
            break
        return "".join(letters) or text

    @staticmethod
    def _monotonic() -> float:
        import time

        return time.perf_counter()
