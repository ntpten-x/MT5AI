from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Mapping, Sequence

import pandas as pd

from invest_advisor_bot.providers.market_data_client import OhlcvBar


class BacktestingEngine:
    """Lightweight candidate backtesting layer with an optional vectorbt dependency."""

    def __init__(
        self,
        *,
        root_dir: Path,
        enabled: bool = False,
        benchmark_ticker: str = "SPY",
        lookback_period: str = "6mo",
        history_limit: int = 126,
        min_history_points: int = 30,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.enabled = bool(enabled)
        self.benchmark_ticker = benchmark_ticker.strip().upper() or "SPY"
        self.lookback_period = lookback_period.strip() or "6mo"
        self.history_limit = max(30, int(history_limit))
        self.min_history_points = max(10, int(min_history_points))
        self._lock = RLock()
        self._warning: str | None = None
        self._last_run_at: str | None = None
        self._backend = "disabled"
        self._last_summary_path = self.root_dir / "candidate_backtests.jsonl"
        if self.enabled:
            self.root_dir.mkdir(parents=True, exist_ok=True)
            self._backend = "vectorbt" if self._vectorbt_available() else "pandas"

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "available": self.enabled,
                "backend": self._backend,
                "benchmark_ticker": self.benchmark_ticker,
                "lookback_period": self.lookback_period,
                "history_limit": self.history_limit,
                "warning": self._warning,
                "last_run_at": self._last_run_at,
            }

    def evaluate_candidate_histories(
        self,
        *,
        candidate_histories: Mapping[str, Sequence[OhlcvBar]],
        benchmark_history: Sequence[OhlcvBar],
    ) -> dict[str, Any]:
        if not self.enabled:
            return {}
        benchmark_series = self._bars_to_close_series(benchmark_history)
        if len(benchmark_series) < self.min_history_points:
            return {}
        benchmark_metrics = self._compute_return_metrics(benchmark_series)
        candidates: list[dict[str, Any]] = []
        for ticker, bars in candidate_histories.items():
            series = self._bars_to_close_series(bars)
            if len(series) < self.min_history_points:
                continue
            metrics = self._compute_return_metrics(series)
            total_return = metrics.get("total_return_pct")
            benchmark_return = benchmark_metrics.get("total_return_pct")
            alpha_pct = None
            if total_return is not None and benchmark_return is not None:
                alpha_pct = round(total_return - benchmark_return, 2)
            candidates.append(
                {
                    "ticker": str(ticker or "").strip().upper(),
                    "total_return_pct": total_return,
                    "alpha_pct": alpha_pct,
                    "max_drawdown_pct": metrics.get("max_drawdown_pct"),
                    "daily_win_rate_pct": metrics.get("daily_win_rate_pct"),
                    "sharpe_like": metrics.get("sharpe_like"),
                    "sample_count": len(series),
                }
            )
        if not candidates:
            return {}
        candidates.sort(
            key=lambda item: (
                float(item.get("alpha_pct") if item.get("alpha_pct") is not None else -9999.0),
                float(item.get("total_return_pct") if item.get("total_return_pct") is not None else -9999.0),
                float(item.get("daily_win_rate_pct") if item.get("daily_win_rate_pct") is not None else -9999.0),
                str(item.get("ticker") or ""),
            ),
            reverse=True,
        )
        alpha_values = [float(item["alpha_pct"]) for item in candidates if item.get("alpha_pct") is not None]
        return_values = [float(item["total_return_pct"]) for item in candidates if item.get("total_return_pct") is not None]
        summary = {
            "generated_at": self._utc_now().isoformat(),
            "benchmark_ticker": self.benchmark_ticker,
            "benchmark_return_pct": benchmark_metrics.get("total_return_pct"),
            "benchmark_max_drawdown_pct": benchmark_metrics.get("max_drawdown_pct"),
            "candidate_count": len(candidates),
            "avg_candidate_return_pct": round(sum(return_values) / len(return_values), 2) if return_values else None,
            "avg_candidate_alpha_pct": round(sum(alpha_values) / len(alpha_values), 2) if alpha_values else None,
            "best_candidate": dict(candidates[0]),
            "candidates": candidates[:5],
        }
        self._append_summary(summary)
        return summary

    def _append_summary(self, summary: Mapping[str, Any]) -> None:
        try:
            with self._last_summary_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(dict(summary), ensure_ascii=False))
                handle.write("\n")
        except OSError as exc:
            with self._lock:
                self._warning = f"backtest_write_failed: {exc}"
        else:
            with self._lock:
                self._last_run_at = self._utc_now().isoformat()
                if self._backend == "disabled":
                    self._backend = "pandas"

    @staticmethod
    def _bars_to_close_series(bars: Sequence[OhlcvBar]) -> pd.Series:
        rows = sorted(
            (
                (item.timestamp, item.close)
                for item in bars
                if item.timestamp is not None and item.close is not None
            ),
            key=lambda item: item[0],
        )
        if not rows:
            return pd.Series(dtype="float64")
        index = pd.DatetimeIndex([item[0] for item in rows])
        values = [float(item[1]) for item in rows]
        series = pd.Series(values, index=index, dtype="float64")
        return series[~series.index.duplicated(keep="last")]

    @staticmethod
    def _compute_return_metrics(close_series: pd.Series) -> dict[str, float | None]:
        if close_series.empty:
            return {
                "total_return_pct": None,
                "max_drawdown_pct": None,
                "daily_win_rate_pct": None,
                "sharpe_like": None,
            }
        returns = close_series.pct_change().dropna()
        first_price = float(close_series.iloc[0])
        last_price = float(close_series.iloc[-1])
        total_return_pct = round(((last_price / first_price) - 1.0) * 100.0, 2) if first_price > 0 else None
        if returns.empty:
            return {
                "total_return_pct": total_return_pct,
                "max_drawdown_pct": None,
                "daily_win_rate_pct": None,
                "sharpe_like": None,
            }
        equity_curve = (1.0 + returns).cumprod()
        rolling_peak = equity_curve.cummax()
        drawdown = (equity_curve / rolling_peak) - 1.0
        std = float(returns.std())
        sharpe_like = None
        if std > 0:
            sharpe_like = round(float(returns.mean()) / std * math.sqrt(252.0), 2)
        return {
            "total_return_pct": total_return_pct,
            "max_drawdown_pct": round(float(drawdown.min()) * 100.0, 2),
            "daily_win_rate_pct": round(float((returns > 0).mean()) * 100.0, 1),
            "sharpe_like": sharpe_like,
        }

    @staticmethod
    def _vectorbt_available() -> bool:
        try:
            import vectorbt  # noqa: F401
        except Exception:
            return False
        return True

    @staticmethod
    def _utc_now() -> datetime:
        return datetime.now(timezone.utc)
