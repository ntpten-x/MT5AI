from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
from datetime import datetime, time, timedelta, timezone
import math
from pathlib import Path
from time import perf_counter
import re
from typing import Mapping

import httpx
from loguru import logger
from telegram.ext import Application, ContextTypes

from invest_advisor_bot.bot.handlers import BOT_SERVICES_KEY, BotServices
from invest_advisor_bot.observability import log_event
from invest_advisor_bot.providers.market_data_client import OhlcvBar
from invest_advisor_bot.providers.live_market_stream import LiveMarketEvent
from invest_advisor_bot.runtime_diagnostics import diagnostics
from invest_advisor_bot.runtime_status import collect_runtime_snapshot
from invest_advisor_bot.services.recommendation_service import RecommendationService

HEALTH_ALERT_STATE_KEY = "health_alert_state"


async def send_daily_digest(context: ContextTypes.DEFAULT_TYPE) -> None:
    services = _get_services(context)
    if services is None or not services.telegram_report_chat_id:
        return

    started_at = perf_counter()
    status = "ok"
    detail: dict[str, object] = {}
    portfolio_holdings = (
        services.portfolio_state_store.list_holdings(services.telegram_report_chat_id)
        if services.portfolio_state_store is not None
        else ()
    )
    try:
        result = await services.recommendation_service.generate_daily_digest(
            news_client=services.news_client,
            market_data_client=services.market_data_client,
            research_client=services.research_client,
            news_limit=services.market_news_limit,
            history_period=services.market_history_period,
            history_interval=services.market_history_interval,
            history_limit=services.market_history_limit,
            portfolio_holdings=portfolio_holdings,
        )
        rendered_text = await _append_ai_portfolio_summary(services, result.recommendation_text, report_kind=report_kind)
        for chunk in _chunk_text(rendered_text, limit=3900):
            await context.bot.send_message(chat_id=services.telegram_report_chat_id, text=chunk)
        detail = {
            "report_kind": "daily_digest",
            "fallback_used": result.fallback_used,
            "model": result.model,
        }
        _record_sent_report(services, "daily_digest", services.telegram_report_chat_id, result)
    except Exception as exc:
        status = "error"
        detail = {"error": str(exc)}
        logger.exception("Failed to send daily digest: {}", exc)
    finally:
        duration_ms = int((perf_counter() - started_at) * 1000)
        diagnostics.record_job_run(job="daily_digest", status=status, duration_ms=duration_ms, detail=detail)
        log_event("job_run", job="daily_digest", status=status, duration_ms=int((perf_counter() - started_at) * 1000))


async def send_morning_report(context: ContextTypes.DEFAULT_TYPE) -> None:
    await _send_periodic_report(context, report_kind="morning")


async def send_midday_report(context: ContextTypes.DEFAULT_TYPE) -> None:
    await _send_periodic_report(context, report_kind="midday")


async def send_closing_report(context: ContextTypes.DEFAULT_TYPE) -> None:
    await _send_periodic_report(context, report_kind="closing")


async def monitor_risk_alerts(context: ContextTypes.DEFAULT_TYPE) -> None:
    services = _get_services(context)
    if services is None or not services.telegram_report_chat_id:
        return

    started_at = perf_counter()
    status = "ok"
    alert_count = 0
    alerts = []
    detail: dict[str, object] = {}
    try:
        alerts = await services.recommendation_service.generate_interest_alerts(
            news_client=services.news_client,
            market_data_client=services.market_data_client,
            research_client=services.research_client,
            news_limit=min(services.market_news_limit, 5),
            history_period=services.market_history_period,
            history_interval=services.market_history_interval,
            history_limit=min(services.market_history_limit, 90),
            vix_threshold=services.risk_vix_alert_threshold,
            risk_score_threshold=services.risk_score_alert_threshold,
            opportunity_score_threshold=services.opportunity_score_alert_threshold,
            news_impact_threshold=services.news_impact_alert_threshold,
        )
        alerts = _filter_active_alerts(alerts)
        if services.alert_state_store is not None:
            alerts = _filter_unsuppressed_alerts(services, alerts)
        alert_count = len(alerts)
        _record_stock_pick_scorecards_from_alerts(services, chat_id=services.telegram_report_chat_id, alerts=alerts)
        for alert in alerts:
            await context.bot.send_message(chat_id=services.telegram_report_chat_id, text=alert.text)
        _record_alert_audits(services, chat_id=services.telegram_report_chat_id, alerts=alerts)
        detail = {
            "alert_count": alert_count,
            "alert_categories": _summarize_alert_categories_map(alerts),
        }
    except Exception as exc:
        status = "error"
        detail = {"error": str(exc)}
        logger.exception("Risk alert monitor failed: {}", exc)
    finally:
        duration_ms = int((perf_counter() - started_at) * 1000)
        diagnostics.record_job_run(job="risk_alert_monitor", status=status, duration_ms=duration_ms, detail=detail)
        if status == "ok" and alert_count:
            diagnostics.record_alert_counts(categories=_summarize_alert_categories_map(alerts))
        log_event(
            "job_run",
            job="risk_alert_monitor",
            status=status,
            duration_ms=int((perf_counter() - started_at) * 1000),
            alert_count=alert_count,
            alert_categories=_summarize_alert_categories(alerts if status == "ok" else []),
        )


async def monitor_stock_pick_alerts(context: ContextTypes.DEFAULT_TYPE) -> None:
    services = _get_services(context)
    if services is None or not services.telegram_report_chat_id:
        return

    prefs = services.user_state_store.get(services.telegram_report_chat_id) if services.user_state_store is not None else None
    preferred_sectors = prefs.preferred_sectors if prefs is not None else ()
    threshold = prefs.stock_alert_threshold if prefs is not None else 1.8
    daily_pick_enabled = prefs.daily_pick_enabled if prefs is not None else True
    approval_mode = prefs.approval_mode if prefs is not None else "auto"
    max_position_size_pct = prefs.max_position_size_pct if prefs is not None else None
    watchlist = prefs.watchlist if prefs is not None else ()
    portfolio_holdings = (
        services.portfolio_state_store.list_holdings(services.telegram_report_chat_id)
        if services.portfolio_state_store is not None
        else ()
    )

    started_at = perf_counter()
    status = "ok"
    alert_count = 0
    alerts = []
    detail: dict[str, object] = {}
    try:
        alerts = await services.recommendation_service.generate_stock_pick_alerts(
            news_client=services.news_client,
            market_data_client=services.market_data_client,
            research_client=services.research_client,
            watchlist=watchlist,
            preferred_sectors=preferred_sectors,
            score_threshold=threshold,
            daily_pick_enabled=daily_pick_enabled,
            limit=5,
            portfolio_holdings=portfolio_holdings,
            approval_mode=approval_mode,
            max_position_size_pct=max_position_size_pct,
        )
        alerts = _filter_active_alerts(alerts)
        if services.alert_state_store is not None:
            alerts = _filter_unsuppressed_alerts(services, alerts)
        alert_count = len(alerts)
        _record_stock_pick_scorecards_from_alerts(services, chat_id=services.telegram_report_chat_id, alerts=alerts)
        for alert in alerts:
            await context.bot.send_message(chat_id=services.telegram_report_chat_id, text=alert.text)
        _record_alert_audits(services, chat_id=services.telegram_report_chat_id, alerts=alerts)
        detail = {
            "alert_count": alert_count,
            "alert_categories": _summarize_alert_categories_map(alerts),
        }
    except Exception as exc:
        status = "error"
        detail = {"error": str(exc)}
        logger.exception("Stock pick monitor failed: {}", exc)
    finally:
        duration_ms = int((perf_counter() - started_at) * 1000)
        diagnostics.record_job_run(job="stock_pick_monitor", status=status, duration_ms=duration_ms, detail=detail)
        if status == "ok" and alert_count:
            diagnostics.record_alert_counts(categories=_summarize_alert_categories_map(alerts))
        log_event(
            "job_run",
            job="stock_pick_monitor",
            status=status,
            duration_ms=int((perf_counter() - started_at) * 1000),
            alert_count=alert_count,
            alert_categories=_summarize_alert_categories(alerts if status == "ok" else []),
        )


async def monitor_sector_and_earnings_alerts(context: ContextTypes.DEFAULT_TYPE) -> None:
    services = _get_services(context)
    if services is None or not services.telegram_report_chat_id:
        return

    prefs = services.user_state_store.get(services.telegram_report_chat_id) if services.user_state_store is not None else None
    watchlist = prefs.watchlist if prefs is not None else ()

    started_at = perf_counter()
    status = "ok"
    alert_count = 0
    alerts = []
    detail: dict[str, object] = {}
    try:
        stock_candidates = await services.recommendation_service._screen_stock_universe(  # type: ignore[attr-defined]
            market_data_client=services.market_data_client,
            top_k=5,
        )
        alerts = await services.recommendation_service.generate_sector_rotation_alerts(
            news_client=services.news_client,
            market_data_client=services.market_data_client,
            research_client=services.research_client,
            sector_rotation_state_store=services.sector_rotation_state_store,
            min_streak=services.sector_rotation_min_streak,
        )
        alerts.extend(
            await services.recommendation_service.generate_earnings_calendar_alerts(
                market_data_client=services.market_data_client,
                watchlist=watchlist,
                top_candidates=stock_candidates,
                days_ahead=max(1, services.earnings_alert_days_ahead),
            )
        )
        alerts.extend(
            await services.recommendation_service.generate_post_earnings_alerts(
                news_client=services.news_client,
                market_data_client=services.market_data_client,
                research_client=services.research_client,
                watchlist=watchlist,
                top_candidates=stock_candidates,
                lookback_days=max(1, services.earnings_result_lookback_days),
            )
        )
        alerts.extend(
            await services.recommendation_service.generate_pre_earnings_risk_alerts(
                market_data_client=services.market_data_client,
                watchlist=watchlist,
                top_candidates=stock_candidates,
                days_ahead=max(1, services.earnings_alert_days_ahead),
            )
        )
        alerts.extend(
            await services.recommendation_service.generate_earnings_setup_alerts(
                market_data_client=services.market_data_client,
                top_candidates=stock_candidates,
                days_ahead=max(3, services.earnings_alert_days_ahead),
            )
        )
        alerts = _filter_active_alerts(alerts)
        if services.alert_state_store is not None:
            alerts = _filter_unsuppressed_alerts(services, alerts)
        alert_count = len(alerts)
        for alert in alerts:
            await context.bot.send_message(chat_id=services.telegram_report_chat_id, text=alert.text)
        _record_alert_audits(services, chat_id=services.telegram_report_chat_id, alerts=alerts)
        detail = {
            "alert_count": alert_count,
            "alert_categories": _summarize_alert_categories_map(alerts),
        }
    except Exception as exc:
        status = "error"
        detail = {"error": str(exc)}
        logger.exception("Sector/earnings monitor failed: {}", exc)
    finally:
        duration_ms = int((perf_counter() - started_at) * 1000)
        diagnostics.record_job_run(job="sector_earnings_monitor", status=status, duration_ms=duration_ms, detail=detail)
        if status == "ok" and alert_count:
            diagnostics.record_alert_counts(categories=_summarize_alert_categories_map(alerts))
        log_event(
            "job_run",
            job="sector_earnings_monitor",
            status=status,
            duration_ms=int((perf_counter() - started_at) * 1000),
            alert_count=alert_count,
            alert_categories=_summarize_alert_categories(alerts if status == "ok" else []),
        )


async def monitor_macro_event_refresh(context: ContextTypes.DEFAULT_TYPE) -> None:
    services = _get_services(context)
    if services is None or not services.telegram_report_chat_id:
        return

    started_at = perf_counter()
    status = "ok"
    alert_count = 0
    alerts = []
    detail: dict[str, object] = {}
    try:
        alerts = await services.recommendation_service.generate_macro_event_driven_alerts(
            market_data_client=services.market_data_client,
            pre_window_minutes=max(1, services.macro_event_pre_window_minutes),
            post_window_minutes=max(10, services.macro_event_post_window_minutes),
            lookahead_hours=max(1, services.macro_event_lookahead_hours),
        )
        alerts = _filter_active_alerts(alerts)
        if services.alert_state_store is not None:
            alerts = _filter_unsuppressed_alerts(services, alerts)
        alert_count = len(alerts)
        for alert in alerts:
            await context.bot.send_message(chat_id=services.telegram_report_chat_id, text=alert.text)
        _record_alert_audits(services, chat_id=services.telegram_report_chat_id, alerts=alerts)
        detail = {
            "alert_count": alert_count,
            "alert_categories": _summarize_alert_categories_map(alerts),
            "pre_window_minutes": services.macro_event_pre_window_minutes,
            "post_window_minutes": services.macro_event_post_window_minutes,
            "lookahead_hours": services.macro_event_lookahead_hours,
        }
    except Exception as exc:
        status = "error"
        detail = {"error": str(exc)}
        logger.exception("Macro event refresh monitor failed: {}", exc)
    finally:
        duration_ms = int((perf_counter() - started_at) * 1000)
        diagnostics.record_job_run(job="macro_event_refresh_monitor", status=status, duration_ms=duration_ms, detail=detail)
        if status == "ok" and alert_count:
            diagnostics.record_alert_counts(categories=_summarize_alert_categories_map(alerts))
        log_event(
            "job_run",
            job="macro_event_refresh_monitor",
            status=status,
            duration_ms=duration_ms,
            alert_count=alert_count,
            alert_categories=_summarize_alert_categories(alerts if status == "ok" else []),
        )


async def evaluate_stock_pick_scorecard(context: ContextTypes.DEFAULT_TYPE) -> None:
    services = _get_services(context)
    if services is None or services.runtime_history_store is None or not services.telegram_report_chat_id:
        return

    started_at = perf_counter()
    status = "ok"
    detail: dict[str, object] = {}
    evaluated_count = 0
    summary_lines: list[str] = []
    try:
        due_rows = services.runtime_history_store.list_due_stock_pick_candidates(limit=20)
        for row in due_rows:
            ticker = str(row.get("ticker") or "").strip().upper()
            entry_price = row.get("entry_price")
            record_id = row.get("id")
            try:
                entry_price_value = float(entry_price)
            except (TypeError, ValueError):
                continue
            if not ticker or entry_price_value <= 0 or record_id is None:
                continue
            quote = await services.market_data_client.get_latest_price(ticker)
            if quote is None or quote.price <= 0:
                continue
            return_pct = (quote.price - entry_price_value) / entry_price_value
            evaluation_detail = await _build_stock_pick_execution_postmortem(
                services=services,
                row=row,
                ticker=ticker,
                entry_price=entry_price_value,
                exit_price=float(quote.price),
                return_pct=float(return_pct),
                outcome_label=_classify_scorecard_outcome(return_pct=float(return_pct)),
            )
            adjusted_return_pct = evaluation_detail.get("return_after_cost_pct")
            try:
                adjusted_return_value = float(adjusted_return_pct)
            except (TypeError, ValueError):
                adjusted_return_value = float(return_pct)
            outcome_label = _classify_scorecard_outcome(return_pct=adjusted_return_value)
            mlflow_run_id = _log_stock_pick_evaluation_to_mlflow(
                services=services,
                row=row,
                ticker=ticker,
                entry_price=entry_price_value,
                exit_price=float(quote.price),
                raw_return_pct=float(return_pct),
                adjusted_return_pct=adjusted_return_value,
                outcome_label=outcome_label,
                evaluation_detail=evaluation_detail,
            )
            _record_stock_pick_evaluation_observers(
                services=services,
                row=row,
                outcome_label=outcome_label,
                adjusted_return_pct=adjusted_return_value,
                evaluation_detail=evaluation_detail,
            )
            if mlflow_run_id:
                evaluation_detail["mlflow_run_id"] = mlflow_run_id
            services.runtime_history_store.complete_stock_pick_evaluation(
                record_id=int(record_id),
                exit_price=float(quote.price),
                return_pct=float(return_pct),
                outcome_label=outcome_label,
                detail=evaluation_detail,
            )
            evaluated_count += 1
            postmortem = str(evaluation_detail.get("postmortem_action") or "-")
            decay_label = str(evaluation_detail.get("signal_decay_label") or "-")
            after_cost_text = (
                f"{adjusted_return_value:+.1%}"
                if adjusted_return_value is not None
                else "-"
            )
            alpha_after_cost_text = ""
            try:
                alpha_after_cost_value = float(evaluation_detail.get("alpha_after_cost_pct"))
                benchmark_label = str(evaluation_detail.get("benchmark_ticker") or evaluation_detail.get("benchmark") or "benchmark")
                alpha_after_cost_text = f" | alpha vs {benchmark_label} {alpha_after_cost_value:+.1%}"
            except (TypeError, ValueError):
                alpha_after_cost_text = ""
            summary_lines.append(
                f"- {ticker}: raw {return_pct:+.1%} | after cost {after_cost_text} | outcome {outcome_label} | "
                f"source {row.get('source_kind') or '-'}{alpha_after_cost_text} | postmortem {postmortem} | decay {decay_label}"
            )

        detail = {"evaluated_count": evaluated_count}
        if summary_lines:
            await context.bot.send_message(
                chat_id=services.telegram_report_chat_id,
                text="Stock Pick Scorecard Update\n" + "\n".join(summary_lines[:8]),
            )
    except Exception as exc:
        status = "error"
        detail = {"error": str(exc), "evaluated_count": evaluated_count}
        logger.exception("Stock pick scorecard evaluation failed: {}", exc)
    finally:
        duration_ms = int((perf_counter() - started_at) * 1000)
        diagnostics.record_job_run(job="stock_pick_scorecard", status=status, duration_ms=duration_ms, detail=detail)
        log_event("job_run", job="stock_pick_scorecard", status=status, duration_ms=duration_ms, **detail)


def _log_stock_pick_evaluation_to_mlflow(
    *,
    services: BotServices,
    row: dict[str, object],
    ticker: str,
    entry_price: float,
    exit_price: float,
    raw_return_pct: float,
    adjusted_return_pct: float,
    outcome_label: str,
    evaluation_detail: dict[str, object],
) -> str | None:
    observer = getattr(services.recommendation_service, "mlflow_observer", None)
    log_evaluation = getattr(observer, "log_evaluation", None)
    if not callable(log_evaluation):
        return None
    artifact_key = str(row.get("recommendation_key") or "").strip()
    record_id = row.get("id")
    tags = {
        "service": "stock_pick_scorecard",
        "artifact_key": artifact_key,
        "scorecard_record_id": str(record_id) if record_id is not None else "",
        "chat_id_hash": _hash_identifier(row.get("chat_id")),
        "ticker": ticker,
        "source_kind": str(row.get("source_kind") or "").strip(),
        "outcome_label": outcome_label,
        "postmortem_action": str(evaluation_detail.get("postmortem_action") or "").strip(),
        "signal_decay_label": str(evaluation_detail.get("signal_decay_label") or "").strip(),
        "benchmark_ticker": str(evaluation_detail.get("benchmark_ticker") or "").strip(),
    }
    metrics = {
        "entry_price": float(entry_price),
        "exit_price": float(exit_price),
        "raw_return_pct": float(raw_return_pct),
        "return_after_cost_pct": float(adjusted_return_pct),
        "alpha_after_cost_pct": _coerce_float(evaluation_detail.get("alpha_after_cost_pct")),
        "alpha_after_cost_vs_sector_pct": _coerce_float(evaluation_detail.get("alpha_after_cost_vs_sector_pct")),
        "alpha_after_cost_vs_peer_pct": _coerce_float(evaluation_detail.get("alpha_after_cost_vs_peer_pct")),
        "benchmark_return_pct": _coerce_float(evaluation_detail.get("benchmark_return_pct")),
        "sector_benchmark_return_pct": _coerce_float(evaluation_detail.get("sector_benchmark_return_pct")),
        "peer_benchmark_return_pct": _coerce_float(evaluation_detail.get("peer_benchmark_return_pct")),
        "ttl_return_pct": _coerce_float(evaluation_detail.get("ttl_return_pct")),
        "ttl_return_after_cost_pct": _coerce_float(evaluation_detail.get("ttl_return_after_cost_pct")),
        "ttl_hit": evaluation_detail.get("ttl_hit"),
        "expired_before_evaluation": evaluation_detail.get("expired_before_evaluation"),
        "execution_cost_bps": _coerce_float(evaluation_detail.get("execution_cost_bps")),
    }
    artifacts = {
        "scorecard_row": row,
        "scorecard_detail": evaluation_detail,
    }
    return log_evaluation(
        name="stock_pick_scorecard_evaluation",
        metrics=metrics,
        artifacts=artifacts,
        tags=tags,
    )


def _record_stock_pick_evaluation_observers(
    *,
    services: BotServices,
    row: dict[str, object],
    outcome_label: str,
    adjusted_return_pct: float,
    evaluation_detail: Mapping[str, object],
) -> None:
    recommendation_service = getattr(services, "recommendation_service", None)
    analytics_store = getattr(recommendation_service, "analytics_store", None)
    if analytics_store is not None:
        try:
            analytics_store.record_evaluation_event(
                artifact_key=str(row.get("recommendation_key") or "").strip(),
                ticker=str(row.get("ticker") or "").strip().upper() or None,
                outcome_label=outcome_label,
                adjusted_return_pct=adjusted_return_pct,
                detail=evaluation_detail,
            )
            analytics_store.record_runtime_snapshot({"runtime": diagnostics.snapshot()})
        except Exception as exc:
            logger.warning("Analytics evaluation logging failed: {}", exc)
    analytics_warehouse = getattr(recommendation_service, "analytics_warehouse", None)
    if analytics_warehouse is not None:
        try:
            analytics_warehouse.record_evaluation_event(
                artifact_key=str(row.get("recommendation_key") or "").strip(),
                ticker=str(row.get("ticker") or "").strip().upper() or None,
                outcome_label=outcome_label,
                adjusted_return_pct=adjusted_return_pct,
                detail=evaluation_detail,
            )
            analytics_warehouse.record_runtime_snapshot({"runtime": diagnostics.snapshot()})
        except Exception as exc:
            logger.warning("Analytics warehouse evaluation logging failed: {}", exc)
    feature_store = getattr(recommendation_service, "feature_store", None)
    if feature_store is not None:
        try:
            feature_store.record_outcome_features(
                artifact_key=str(row.get("recommendation_key") or "").strip(),
                outcome_label=outcome_label,
                adjusted_return_pct=adjusted_return_pct,
                detail=evaluation_detail,
            )
        except Exception as exc:
            logger.warning("Feature store outcome logging failed: {}", exc)
    hot_path_cache = getattr(recommendation_service, "hot_path_cache", None)
    if hot_path_cache is not None:
        try:
            payload = {
                "artifact_key": str(row.get("recommendation_key") or "").strip(),
                "ticker": str(row.get("ticker") or "").strip().upper() or None,
                "outcome_label": outcome_label,
                "adjusted_return_pct": adjusted_return_pct,
            }
            hot_path_cache.set_json(namespace="evaluation", key=str(row.get("recommendation_key") or "").strip(), payload=payload, ttl_seconds=1800)
            hot_path_cache.append_stream(stream="evaluations", payload=payload)
        except Exception as exc:
            logger.warning("Hot-path cache evaluation logging failed: {}", exc)
    event_bus = getattr(recommendation_service, "event_bus", None)
    if event_bus is not None:
        try:
            event_bus.publish(
                topic="evaluation_event",
                key=str(row.get("recommendation_key") or "").strip(),
                payload={
                    "ticker": str(row.get("ticker") or "").strip().upper() or None,
                    "outcome_label": outcome_label,
                    "adjusted_return_pct": adjusted_return_pct,
                },
            )
        except Exception as exc:
            logger.warning("Event bus evaluation publish failed: {}", exc)
    evidently_observer = getattr(recommendation_service, "evidently_observer", None)
    if evidently_observer is not None:
        try:
            evidently_observer.log_outcome(
                artifact_key=str(row.get("recommendation_key") or "").strip(),
                outcome_label=outcome_label,
                return_after_cost_pct=adjusted_return_pct,
                detail=evaluation_detail,
            )
        except Exception as exc:
            logger.warning("Evidently outcome logging failed: {}", exc)
    braintrust_observer = getattr(recommendation_service, "braintrust_observer", None)
    if braintrust_observer is not None:
        try:
            braintrust_observer.log_outcome(
                artifact_key=str(row.get("recommendation_key") or "").strip(),
                outcome_label=outcome_label,
                return_after_cost_pct=adjusted_return_pct,
                detail=evaluation_detail,
            )
        except Exception as exc:
            logger.warning("Braintrust outcome logging failed: {}", exc)
    langfuse_observer = getattr(recommendation_service, "langfuse_observer", None)
    if langfuse_observer is not None:
        try:
            langfuse_observer.log_outcome(
                artifact_key=str(row.get("recommendation_key") or "").strip(),
                outcome_label=outcome_label,
                return_after_cost_pct=adjusted_return_pct,
                detail=evaluation_detail,
            )
        except Exception as exc:
            logger.warning("Langfuse outcome logging failed: {}", exc)


async def monitor_live_market_stream(context: ContextTypes.DEFAULT_TYPE) -> None:
    services = _get_services(context)
    if services is None:
        return
    stream_client = getattr(services, "live_market_stream_client", None)
    if stream_client is None or not stream_client.available():
        return

    started_at = perf_counter()
    status = "ok"
    detail: dict[str, object] = {}
    events: list[LiveMarketEvent] = []
    try:
        symbols = tuple(services.live_stream_symbols or ())
        events = await stream_client.sample_events(symbols)
        recommendation_service = getattr(services, "recommendation_service", None)
        analytics_store = getattr(recommendation_service, "analytics_store", None)
        analytics_warehouse = getattr(recommendation_service, "analytics_warehouse", None)
        hot_path_cache = getattr(recommendation_service, "hot_path_cache", None)
        event_bus = getattr(recommendation_service, "event_bus", None)
        if analytics_store is not None:
            for event in events[: max(1, services.live_stream_max_events)]:
                analytics_store.record_market_event(
                    artifact_key=f"live_stream::{event.symbol}::{event.captured_at.isoformat()}",
                    topic="live_market_stream",
                    detail={
                        "symbol": event.symbol,
                        "dataset": event.dataset,
                        "schema": event.schema,
                        "event_type": event.event_type,
                        "price": event.price,
                        "size": event.size,
                        "bid": event.bid,
                        "ask": event.ask,
                        "spread_bps": event.spread_bps,
                        "captured_at": event.captured_at.isoformat(),
                    },
                    numeric_1=event.spread_bps,
                    numeric_2=event.price,
                )
        if analytics_warehouse is not None:
            for event in events[: max(1, services.live_stream_max_events)]:
                analytics_warehouse.record_market_event(
                    artifact_key=f"live_stream::{event.symbol}::{event.captured_at.isoformat()}",
                    topic="live_market_stream",
                    detail={
                        "symbol": event.symbol,
                        "dataset": event.dataset,
                        "schema": event.schema,
                        "event_type": event.event_type,
                        "price": event.price,
                        "size": event.size,
                        "bid": event.bid,
                        "ask": event.ask,
                        "spread_bps": event.spread_bps,
                        "captured_at": event.captured_at.isoformat(),
                    },
                    numeric_1=event.spread_bps,
                    numeric_2=event.price,
                )
        if hot_path_cache is not None:
            for event in events[: max(1, services.live_stream_max_events)]:
                hot_path_cache.append_stream(
                    stream="market_events",
                    payload={
                        "symbol": event.symbol,
                        "dataset": event.dataset,
                        "schema": event.schema,
                        "spread_bps": event.spread_bps,
                        "price": event.price,
                    },
                )
        if event_bus is not None:
            for event in events[: max(1, services.live_stream_max_events)]:
                event_bus.publish(
                    topic="market_event",
                    key=event.symbol,
                    payload={
                        "symbol": event.symbol,
                        "dataset": event.dataset,
                        "schema": event.schema,
                        "spread_bps": event.spread_bps,
                        "price": event.price,
                    },
                )
        spread_alerts = [
            event
            for event in events
            if event.spread_bps is not None and event.spread_bps >= services.live_stream_spread_alert_bps
        ]
        if spread_alerts and services.telegram_report_chat_id:
            lines = ["Live Spread Alert"]
            for event in spread_alerts[:4]:
                lines.append(
                    f"- {event.symbol}: spread={event.spread_bps} bps | price={event.price if event.price is not None else '-'} | schema={event.schema}"
                )
            await context.bot.send_message(chat_id=services.telegram_report_chat_id, text="\n".join(lines))
        detail = {
            "event_count": len(events),
            "symbols": list(symbols),
            "spread_alert_count": len(spread_alerts),
        }
    except Exception as exc:
        status = "error"
        detail = {"error": str(exc)}
        logger.exception("Live market stream monitor failed: {}", exc)
    finally:
        duration_ms = int((perf_counter() - started_at) * 1000)
        diagnostics.record_job_run(job="live_market_stream_monitor", status=status, duration_ms=duration_ms, detail=detail)
        log_event("job_run", job="live_market_stream_monitor", status=status, duration_ms=duration_ms, **detail)


async def consume_event_bus_worker(context: ContextTypes.DEFAULT_TYPE) -> None:
    services = _get_services(context)
    if services is None:
        return
    recommendation_service = services.recommendation_service
    worker = getattr(recommendation_service, "event_bus_consumer", None)
    if worker is None:
        return

    started_at = perf_counter()
    status = "ok"
    detail: dict[str, object] = {}
    try:
        result = await asyncio.to_thread(worker.process_pending)
        detail = dict(result)
    except Exception as exc:
        status = "error"
        detail = {"error": str(exc)}
        logger.exception("Event bus consumer worker failed: {}", exc)
    finally:
        duration_ms = int((perf_counter() - started_at) * 1000)
        diagnostics.record_job_run(job="event_bus_consumer", status=status, duration_ms=duration_ms, detail=detail)
        log_event("job_run", job="event_bus_consumer", status=status, duration_ms=duration_ms, **detail)


async def run_runtime_maintenance(context: ContextTypes.DEFAULT_TYPE) -> None:
    services = _get_services(context)
    if services is None:
        return

    started_at = perf_counter()
    status = "ok"
    detail: dict[str, object] = {}
    try:
        if services.runtime_history_store is not None:
            services.runtime_history_store.cleanup_retention()
        if services.sector_rotation_state_store is not None:
            services.sector_rotation_state_store.cleanup_retention()
        deleted_logs = _cleanup_old_logs(services.logs_dir, services.log_retention)
        deleted_backups = 0
        if services.backup_manager is not None and services.backup_manager.available():
            deleted_backups = services.backup_manager.cleanup_retention()
        detail = {"deleted_logs": deleted_logs, "deleted_backups": deleted_backups}
    except Exception as exc:
        status = "error"
        detail = {"error": str(exc)}
        logger.exception("Runtime maintenance failed: {}", exc)
    finally:
        duration_ms = int((perf_counter() - started_at) * 1000)
        diagnostics.record_job_run(job="runtime_maintenance", status=status, duration_ms=duration_ms, detail=detail)
        log_event("job_run", job="runtime_maintenance", status=status, duration_ms=duration_ms, **detail)


async def run_backup_snapshot(context: ContextTypes.DEFAULT_TYPE) -> None:
    services = _get_services(context)
    if services is None or services.backup_manager is None or not services.backup_manager.available():
        return

    started_at = perf_counter()
    status = "ok"
    detail: dict[str, object] = {}
    try:
        manifest = await asyncio.to_thread(services.backup_manager.create_backup, reason="scheduled")
        detail = {
            "path": str(manifest.path),
            "table_count": len(manifest.row_counts),
        }
    except Exception as exc:
        status = "error"
        detail = {"error": str(exc)}
        logger.exception("Scheduled backup failed: {}", exc)
    finally:
        duration_ms = int((perf_counter() - started_at) * 1000)
        diagnostics.record_job_run(job="scheduled_backup", status=status, duration_ms=duration_ms, detail=detail)
        log_event("job_run", job="scheduled_backup", status=status, duration_ms=duration_ms, **detail)


async def monitor_health_webhook(context: ContextTypes.DEFAULT_TYPE) -> None:
    services = _get_services(context)
    if services is None or not services.health_alert_webhook_url.strip():
        return

    started_at = perf_counter()
    status = "ok"
    detail: dict[str, object] = {}
    sent_event: str | None = None
    issue_count = 0
    try:
        snapshot = collect_runtime_snapshot(services, ping_database=True)
        issues = _build_health_alert_issues(snapshot)
        issue_count = len(issues)
        event_type = _resolve_health_event_type(context=context, services=services, issues=issues)
        if event_type is not None:
            delivery_sequence = _next_health_delivery_sequence(context=context)
            incident_key = _build_health_issue_signature(issues)
            idempotency_key = _build_health_idempotency_key(
                event_type=event_type,
                issues=issues,
                delivery_sequence=delivery_sequence,
            )
            await _post_health_alert_webhook(
                url=services.health_alert_webhook_url,
                secret=services.health_alert_webhook_secret,
                timeout_seconds=services.health_alert_timeout_seconds,
                retry_count=services.health_alert_retry_count,
                retry_backoff_seconds=services.health_alert_retry_backoff_seconds,
                idempotency_key=idempotency_key,
                payload=_build_health_alert_payload(
                    snapshot=snapshot,
                    issues=issues,
                    event_type=event_type,
                    idempotency_key=idempotency_key,
                    incident_key=incident_key,
                ),
            )
            _record_health_alert_state(
                context=context,
                issues=issues,
                event_type=event_type,
                delivery_sequence=delivery_sequence,
            )
            sent_event = event_type
        detail = {
            "issue_count": issue_count,
            "sent_event": sent_event,
            "issue_kinds": [str(item.get("kind") or "") for item in issues],
        }
    except Exception as exc:
        status = "error"
        detail = {"error": str(exc), "issue_count": issue_count, "sent_event": sent_event}
        logger.exception("Health webhook monitor failed: {}", exc)
    finally:
        duration_ms = int((perf_counter() - started_at) * 1000)
        diagnostics.record_job_run(job="health_webhook_monitor", status=status, duration_ms=duration_ms, detail=detail)
        log_event(
            "job_run",
            job="health_webhook_monitor",
            status=status,
            duration_ms=duration_ms,
            issue_count=issue_count,
            sent_event=sent_event,
        )


async def rebalance_ai_simulated_portfolio(context: ContextTypes.DEFAULT_TYPE) -> None:
    services = _get_services(context)
    if services is None or not services.telegram_report_chat_id or services.ai_simulated_portfolio_service is None:
        return

    started_at = perf_counter()
    status = "ok"
    detail: dict[str, object] = {}
    try:
        prior_state = services.ai_simulated_portfolio_service.ensure_portfolio(services.telegram_report_chat_id)
        result = await services.ai_simulated_portfolio_service.maybe_rebalance(
            conversation_key=services.telegram_report_chat_id,
            reason="scheduled_job",
            force=False,
        )
        detail = {
            "action_count": result.action_count,
            "skipped_reason": result.skipped_reason,
            "total_value": result.snapshot.get("total_value"),
        }
        if result.skipped_reason != "cooldown_active" and (
            result.action_count > 0 or prior_state.last_rebalanced_at is None
        ):
            alert_texts = services.ai_simulated_portfolio_service.render_trade_alert_texts(
                snapshot=result.snapshot,
                trades=result.trades,
            )
            for alert_text in alert_texts:
                await context.bot.send_message(chat_id=services.telegram_report_chat_id, text=alert_text)
            for chunk in _chunk_text(result.rendered_summary, limit=3900):
                await context.bot.send_message(chat_id=services.telegram_report_chat_id, text=chunk)
    except Exception as exc:
        status = "error"
        detail = {"error": str(exc)}
        logger.exception("AI simulated portfolio rebalance failed: {}", exc)
    finally:
        duration_ms = int((perf_counter() - started_at) * 1000)
        diagnostics.record_job_run(job="ai_simulated_portfolio", status=status, duration_ms=duration_ms, detail=detail)
        log_event("job_run", job="ai_simulated_portfolio", status=status, duration_ms=duration_ms, **detail)


def register_jobs(
    application: Application,
    *,
    daily_digest_hour_utc: int,
    daily_digest_minute_utc: int,
    morning_report_hour_utc: int,
    morning_report_minute_utc: int,
    midday_report_hour_utc: int,
    midday_report_minute_utc: int,
    closing_report_hour_utc: int,
    closing_report_minute_utc: int,
    risk_check_interval_minutes: int,
    macro_event_refresh_interval_minutes: int,
    earnings_alert_days_ahead: int,
    maintenance_cleanup_interval_minutes: int,
    stock_pick_evaluation_interval_minutes: int,
    ai_simulated_portfolio_rebalance_interval_minutes: int,
    backup_interval_hours: int,
    health_alert_interval_minutes: int,
    live_stream_poll_interval_seconds: int,
    event_bus_consumer_poll_interval_seconds: int,
    health_alert_only: bool = False,
) -> None:
    jq = application.job_queue
    if jq is None:
        logger.warning("Job queue is not available; scheduled jobs will be skipped")
        return

    jq.run_repeating(
        monitor_health_webhook,
        interval=timedelta(minutes=max(1, health_alert_interval_minutes)),
        first=15,
        name="health-webhook-monitor",
    )

    if health_alert_only:
        logger.info("Telegram health monitoring job registered")
        return

    jq.run_repeating(
        monitor_live_market_stream,
        interval=timedelta(seconds=max(15, live_stream_poll_interval_seconds)),
        first=12,
        name="live-market-stream-monitor",
    )
    jq.run_repeating(
        consume_event_bus_worker,
        interval=timedelta(seconds=max(15, event_bus_consumer_poll_interval_seconds)),
        first=18,
        name="event-bus-consumer",
    )

    jq.run_daily(
        send_daily_digest,
        time=time(hour=daily_digest_hour_utc, minute=daily_digest_minute_utc, tzinfo=timezone.utc),
        name="daily-digest",
    )
    jq.run_daily(
        send_morning_report,
        time=time(hour=morning_report_hour_utc, minute=morning_report_minute_utc, tzinfo=timezone.utc),
        name="morning-report",
    )
    jq.run_daily(
        send_midday_report,
        time=time(hour=midday_report_hour_utc, minute=midday_report_minute_utc, tzinfo=timezone.utc),
        name="midday-report",
    )
    jq.run_daily(
        send_closing_report,
        time=time(hour=closing_report_hour_utc, minute=closing_report_minute_utc, tzinfo=timezone.utc),
        name="closing-report",
    )
    jq.run_repeating(
        monitor_risk_alerts,
        interval=timedelta(minutes=max(1, risk_check_interval_minutes)),
        first=10,
        name="risk-alert-monitor",
    )
    jq.run_repeating(
        monitor_stock_pick_alerts,
        interval=timedelta(minutes=max(5, risk_check_interval_minutes)),
        first=20,
        name="stock-pick-monitor",
    )
    jq.run_repeating(
        monitor_macro_event_refresh,
        interval=timedelta(minutes=max(1, macro_event_refresh_interval_minutes)),
        first=25,
        name="macro-event-refresh-monitor",
    )
    jq.run_repeating(
        monitor_sector_and_earnings_alerts,
        interval=timedelta(minutes=max(10, risk_check_interval_minutes)),
        first=30,
        name="sector-earnings-monitor",
    )
    jq.run_repeating(
        run_runtime_maintenance,
        interval=timedelta(minutes=max(60, maintenance_cleanup_interval_minutes)),
        first=45,
        name="runtime-maintenance",
    )
    jq.run_repeating(
        evaluate_stock_pick_scorecard,
        interval=timedelta(minutes=max(60, stock_pick_evaluation_interval_minutes)),
        first=55,
        name="stock-pick-scorecard",
    )
    jq.run_repeating(
        rebalance_ai_simulated_portfolio,
        interval=timedelta(minutes=max(60, ai_simulated_portfolio_rebalance_interval_minutes)),
        first=75,
        name="ai-simulated-portfolio",
    )
    jq.run_repeating(
        run_backup_snapshot,
        interval=timedelta(hours=max(6, backup_interval_hours)),
        first=65,
        name="scheduled-backup",
    )
    logger.info("Telegram scheduled jobs registered")


def _get_services(context: ContextTypes.DEFAULT_TYPE) -> BotServices | None:
    services = context.application.bot_data.get(BOT_SERVICES_KEY)
    if isinstance(services, BotServices):
        return services
    return None


def _build_health_alert_issues(snapshot: dict[str, object]) -> list[dict[str, object]]:
    issues: list[dict[str, object]] = []

    db_state = snapshot.get("db_state")
    if isinstance(db_state, dict) and db_state.get("healthy") is False:
        issues.append(
            {
                "kind": "db_unhealthy",
                "severity": "critical",
                "summary": f"database backend {db_state.get('backend') or 'unknown'} is unhealthy",
                "detail": {"backend": db_state.get("backend"), "error": db_state.get("error")},
            }
        )

    provider_circuit = snapshot.get("provider_circuit")
    if isinstance(provider_circuit, dict):
        open_providers = sorted(
            str(provider)
            for provider, state in provider_circuit.items()
            if isinstance(state, dict) and bool(state.get("is_open"))
        )
        if open_providers:
            issues.append(
                {
                    "kind": "provider_circuit_open",
                    "severity": "warning",
                    "summary": f"provider circuits open: {', '.join(open_providers)}",
                    "detail": {"providers": open_providers},
                }
            )

    mlflow_state = snapshot.get("mlflow")
    if isinstance(mlflow_state, dict):
        warning = str(mlflow_state.get("warning") or "").strip()
        if warning:
            issues.append(
                {
                    "kind": "mlflow_warning",
                    "severity": "warning",
                    "summary": warning,
                    "detail": {"warning": warning, "enabled": bool(mlflow_state.get("enabled"))},
                }
            )
    return issues


def _resolve_health_event_type(
    *,
    context: ContextTypes.DEFAULT_TYPE,
    services: BotServices,
    issues: list[dict[str, object]],
) -> str | None:
    state = context.application.bot_data.get(HEALTH_ALERT_STATE_KEY)
    if not isinstance(state, dict):
        state = {}
    previous_signature = str(state.get("signature") or "")
    previous_event = str(state.get("event_type") or "")
    last_sent_at = _parse_iso_datetime(state.get("last_sent_at"))
    now = datetime.now(timezone.utc)
    cooldown = timedelta(minutes=max(1, services.health_alert_cooldown_minutes))
    current_signature = _build_health_issue_signature(issues)

    if issues:
        if current_signature != previous_signature or previous_event != "incident":
            return "incident"
        if last_sent_at is None or now - last_sent_at >= cooldown:
            return "incident"
        return None

    if previous_event == "incident":
        return "resolved"
    return None


def _next_health_delivery_sequence(*, context: ContextTypes.DEFAULT_TYPE) -> int:
    state = context.application.bot_data.get(HEALTH_ALERT_STATE_KEY)
    if not isinstance(state, dict):
        return 1
    return int(state.get("delivery_sequence") or 0) + 1


def _record_health_alert_state(
    *,
    context: ContextTypes.DEFAULT_TYPE,
    issues: list[dict[str, object]],
    event_type: str,
    delivery_sequence: int,
) -> None:
    context.application.bot_data[HEALTH_ALERT_STATE_KEY] = {
        "signature": _build_health_issue_signature(issues),
        "event_type": event_type,
        "last_sent_at": datetime.now(timezone.utc).isoformat(),
        "delivery_sequence": int(delivery_sequence),
    }


def _build_health_issue_signature(issues: list[dict[str, object]]) -> str:
    if not issues:
        return "healthy"
    parts: list[str] = []
    for issue in issues:
        kind = str(issue.get("kind") or "").strip()
        detail = issue.get("detail")
        if isinstance(detail, dict):
            normalized_detail = "|".join(f"{key}={detail[key]}" for key in sorted(detail))
        else:
            normalized_detail = ""
        parts.append(f"{kind}:{normalized_detail}")
    return ";".join(sorted(parts))


def _build_health_alert_payload(
    *,
    snapshot: dict[str, object],
    issues: list[dict[str, object]],
    event_type: str,
    idempotency_key: str,
    incident_key: str,
) -> dict[str, object]:
    return {
        "service": "invest-advisor-bot",
        "event_type": event_type,
        "incident_key": incident_key,
        "idempotency_key": idempotency_key,
        "alerted_at": datetime.now(timezone.utc).isoformat(),
        "issue_count": len(issues),
        "issues": issues,
        "summary": _build_health_alert_summary(event_type=event_type, issues=issues),
        "runtime": {
            "uptime_seconds": snapshot.get("uptime_seconds"),
            "started_at": snapshot.get("started_at"),
            "db_state": snapshot.get("db_state"),
            "provider_circuit": snapshot.get("provider_circuit"),
            "mlflow": snapshot.get("mlflow"),
        },
    }


def _build_health_alert_summary(*, event_type: str, issues: list[dict[str, object]]) -> str:
    if event_type == "resolved":
        return "runtime health recovered"
    summaries = [str(issue.get("summary") or "").strip() for issue in issues if str(issue.get("summary") or "").strip()]
    if not summaries:
        return "runtime health incident detected"
    return " | ".join(summaries[:3])


def _build_health_idempotency_key(
    *,
    event_type: str,
    issues: list[dict[str, object]],
    delivery_sequence: int,
) -> str:
    incident_key = _build_health_issue_signature(issues)
    payload = f"{event_type}|{incident_key}|{int(delivery_sequence)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


async def _post_health_alert_webhook(
    *,
    url: str,
    secret: str,
    timeout_seconds: float,
    retry_count: int,
    retry_backoff_seconds: float,
    idempotency_key: str,
    payload: dict[str, object],
) -> None:
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    timestamp = datetime.now(timezone.utc).isoformat()
    headers = {
        "Content-Type": "application/json",
        "X-Invest-Advisor-Idempotency-Key": idempotency_key,
    }
    normalized_secret = secret.strip()
    if normalized_secret:
        signature_payload = timestamp.encode("utf-8") + b"." + body
        digest = hmac.new(normalized_secret.encode("utf-8"), signature_payload, hashlib.sha256).hexdigest()
        headers["X-Invest-Advisor-Timestamp"] = timestamp
        headers["X-Invest-Advisor-Signature"] = f"sha256={digest}"

    max_attempts = max(1, int(retry_count) + 1)
    backoff_base = max(0.1, float(retry_backoff_seconds))
    async with httpx.AsyncClient(timeout=max(1.0, float(timeout_seconds))) as client:
        for attempt in range(max_attempts):
            try:
                response = await client.post(url, content=body, headers=headers)
                response.raise_for_status()
                return
            except httpx.HTTPStatusError as exc:
                if attempt + 1 >= max_attempts or exc.response.status_code not in {408, 409, 425, 429, 500, 502, 503, 504}:
                    raise
            except httpx.HTTPError:
                if attempt + 1 >= max_attempts:
                    raise
            await asyncio.sleep(backoff_base * (2**attempt))


async def _send_periodic_report(context: ContextTypes.DEFAULT_TYPE, *, report_kind: str) -> None:
    services = _get_services(context)
    if services is None or not services.telegram_report_chat_id:
        return
    started_at = perf_counter()
    status = "ok"
    detail: dict[str, object] = {}
    try:
        result = await services.recommendation_service.generate_periodic_report(
            report_kind=report_kind,  # type: ignore[arg-type]
            news_client=services.news_client,
            market_data_client=services.market_data_client,
            research_client=services.research_client,
            sector_rotation_state_store=services.sector_rotation_state_store,
            report_memory_store=services.report_memory_store,
            sector_rotation_min_streak=services.sector_rotation_min_streak,
            earnings_result_lookback_days=services.earnings_result_lookback_days,
            news_limit=services.market_news_limit,
            history_period=services.market_history_period,
            history_interval=services.market_history_interval,
            history_limit=services.market_history_limit,
            portfolio_holdings=(
                services.portfolio_state_store.list_holdings(services.telegram_report_chat_id)
                if services.portfolio_state_store is not None
                else ()
            ),
        )
        rendered_text = await _append_ai_portfolio_summary(services, result.recommendation_text)
        for chunk in _chunk_text(rendered_text, limit=3900):
            await context.bot.send_message(chat_id=services.telegram_report_chat_id, text=chunk)
        _record_sent_report(services, report_kind, services.telegram_report_chat_id, result)
        detail = {
            "report_kind": report_kind,
            "fallback_used": result.fallback_used,
            "model": result.model,
        }
    except Exception as exc:
        status = "error"
        detail = {"error": str(exc)}
        logger.exception("Failed to send {} report: {}", report_kind, exc)
    finally:
        duration_ms = int((perf_counter() - started_at) * 1000)
        diagnostics.record_job_run(job=f"{report_kind}_report", status=status, duration_ms=duration_ms, detail=detail)
        log_event(
            "job_run",
            job=f"{report_kind}_report",
            status=status,
            duration_ms=int((perf_counter() - started_at) * 1000),
        )


async def _append_ai_portfolio_summary(services: BotServices, text: str, *, report_kind: str | None = None) -> str:
    ai_service = services.ai_simulated_portfolio_service
    if ai_service is None:
        return text
    snapshot = await ai_service.build_snapshot(conversation_key=services.telegram_report_chat_id)
    summary = ai_service.render_daily_digest_text(snapshot=snapshot, report_kind=report_kind)
    return f"{text}\n\n{summary}" if summary.strip() else text


def _record_sent_report(services: BotServices, report_kind: str, chat_id: str, result: object) -> None:
    if services.runtime_history_store is None:
        return
    recommendation_text = str(getattr(result, "recommendation_text", "") or "")
    fallback_used = bool(getattr(result, "fallback_used", False))
    model = getattr(result, "model", None)
    detail: dict[str, object] = {"source": "scheduled_job"}
    input_payload = getattr(result, "input_payload", None)
    if isinstance(input_payload, dict):
        coverage = RecommendationService.summarize_source_coverage(input_payload)
        if coverage.get("used_sources"):
            detail["source_coverage"] = coverage
        if isinstance(input_payload.get("source_health"), dict):
            detail["source_health"] = dict(input_payload.get("source_health") or {})
        if isinstance(input_payload.get("champion_challenger"), dict):
            detail["champion_challenger"] = dict(input_payload.get("champion_challenger") or {})
        if isinstance(input_payload.get("no_trade_decision"), dict):
            detail["no_trade_decision"] = dict(input_payload.get("no_trade_decision") or {})
    services.runtime_history_store.record_sent_report(
        report_kind=report_kind,
        chat_id=str(chat_id),
        fallback_used=fallback_used,
        model=str(model) if isinstance(model, str) else None,
        summary=recommendation_text[:240],
        detail=detail,
    )


def _record_alert_audits(services: BotServices, *, chat_id: str, alerts: list[object]) -> None:
    if services.runtime_history_store is None:
        return
    for alert in alerts:
        alert_key = str(getattr(alert, "key", "") or "unknown")
        category = alert_key.split(":", 1)[0] if alert_key else "unknown"
        services.runtime_history_store.record_alert_audit(
            alert_key=alert_key,
            severity=str(getattr(alert, "severity", "info") or "info"),
            category=category,
            chat_id=str(chat_id),
            text_excerpt=str(getattr(alert, "text", "") or "")[:400],
            detail=dict(getattr(alert, "metadata", None) or {}) if isinstance(getattr(alert, "metadata", None), dict) else {},
        )


def _record_stock_pick_scorecards_from_alerts(services: BotServices, *, chat_id: str, alerts: list[object]) -> None:
    if services.runtime_history_store is None:
        return
    due_at = datetime.now(timezone.utc) + timedelta(days=max(1, services.stock_pick_evaluation_horizon_days))
    for alert in alerts:
        metadata = getattr(alert, "metadata", None)
        if not isinstance(metadata, dict) or not metadata.get("stock_pick"):
            continue
        ticker = str(metadata.get("ticker") or "").strip().upper()
        entry_price = metadata.get("entry_price")
        alert_key = str(getattr(alert, "key", "") or "").strip()
        try:
            entry_price_value = float(entry_price)
        except (TypeError, ValueError):
            continue
        if not ticker or not alert_key or entry_price_value <= 0:
            continue
        services.runtime_history_store.record_stock_pick_candidate(
            recommendation_key=alert_key,
            due_at=due_at,
            source_kind=str(metadata.get("source_kind") or "stock_pick_alert"),
            chat_id=str(chat_id),
            ticker=ticker,
            company_name=str(metadata.get("company_name") or ticker),
            stance=str(metadata.get("stance") or "watch"),
            confidence_score=float(metadata.get("confidence_score")) if metadata.get("confidence_score") is not None else None,
            confidence_label=str(metadata.get("confidence_label") or "") or None,
            composite_score=float(metadata.get("composite_score")) if metadata.get("composite_score") is not None else None,
            entry_price=entry_price_value,
            detail={
                "due_at": due_at.isoformat(),
                "source_coverage": metadata.get("source_coverage") if isinstance(metadata.get("source_coverage"), dict) else {},
                "thesis_summary": str(metadata.get("thesis_summary") or "").strip() or None,
                "thesis_memory": metadata.get("thesis_memory") if isinstance(metadata.get("thesis_memory"), list) else [],
                "macro_headline": str(metadata.get("macro_headline") or "").strip() or None,
                "macro_drivers": metadata.get("macro_drivers") if isinstance(metadata.get("macro_drivers"), list) else [],
                "alert_kind": str(metadata.get("alert_kind") or "stock_pick"),
                "position_size_pct": metadata.get("position_size_pct"),
                "position_size_tier": metadata.get("position_size_tier"),
                "ttl_minutes": metadata.get("ttl_minutes"),
                "expires_at": metadata.get("expires_at"),
                "realert_after_minutes": metadata.get("realert_after_minutes"),
                "benchmark": metadata.get("benchmark"),
                "benchmark_ticker": metadata.get("benchmark_ticker"),
                "peer_benchmark_ticker": metadata.get("peer_benchmark_ticker"),
                "sector": metadata.get("sector"),
                "source_health": metadata.get("source_health") if isinstance(metadata.get("source_health"), dict) else {},
                "no_trade_decision": (
                    metadata.get("no_trade_decision") if isinstance(metadata.get("no_trade_decision"), dict) else {}
                ),
                "portfolio_constraints": (
                    metadata.get("portfolio_constraints") if isinstance(metadata.get("portfolio_constraints"), dict) else {}
                ),
                "execution_realism": (
                    metadata.get("execution_realism") if isinstance(metadata.get("execution_realism"), dict) else {}
                ),
            },
        )


def _filter_active_alerts(alerts: list[object]) -> list[object]:
    now = datetime.now(timezone.utc)
    active: list[object] = []
    for alert in alerts:
        metadata = getattr(alert, "metadata", None)
        if not isinstance(metadata, dict):
            active.append(alert)
            continue
        raw_expires_at = metadata.get("expires_at")
        if not isinstance(raw_expires_at, str) or not raw_expires_at.strip():
            active.append(alert)
            continue
        try:
            expires_at = datetime.fromisoformat(raw_expires_at)
        except ValueError:
            active.append(alert)
            continue
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
        if expires_at > now:
            active.append(alert)
    return active


def _filter_unsuppressed_alerts(services: BotServices, alerts: list[object]) -> list[object]:
    if services.alert_state_store is None:
        return alerts
    filter_alerts = getattr(services.alert_state_store, "filter_alerts", None)
    if callable(filter_alerts):
        try:
            filtered = filter_alerts(alerts)
        except Exception:
            filtered = None
        if isinstance(filtered, list):
            return filtered
    allowed_keys = set(services.alert_state_store.filter_new_keys(alert.key for alert in alerts))
    return [alert for alert in alerts if alert.key in allowed_keys]


async def _build_stock_pick_execution_postmortem(
    *,
    services: BotServices,
    row: dict[str, object],
    ticker: str,
    entry_price: float,
    exit_price: float,
    return_pct: float,
    outcome_label: str,
) -> dict[str, object]:
    now = datetime.now(timezone.utc)
    detail = dict(row.get("detail") or {}) if isinstance(row.get("detail"), dict) else {}
    detail["evaluated_at"] = now.isoformat()
    detail["outcome_label"] = outcome_label
    detail["exit_price"] = round(exit_price, 4)
    execution_realism = detail.get("execution_realism")
    try:
        execution_cost_bps = float(execution_realism.get("execution_cost_bps")) if isinstance(execution_realism, dict) else 0.0
    except (TypeError, ValueError):
        execution_cost_bps = 0.0
    return_after_cost_pct = float(return_pct) - (execution_cost_bps / 10000.0)
    detail["execution_cost_bps"] = round(execution_cost_bps, 1)
    detail["return_after_cost_pct"] = round(return_after_cost_pct, 4)
    detail["outcome_label_raw"] = outcome_label
    detail["outcome_label_after_cost"] = _classify_scorecard_outcome(return_pct=return_after_cost_pct)
    benchmark_ticker = _resolve_benchmark_ticker(detail=detail)
    detail["benchmark_ticker"] = benchmark_ticker
    benchmark_return_pct, benchmark_entry_price, benchmark_exit_price = await _load_benchmark_return(
        services=services,
        benchmark_ticker=benchmark_ticker,
        created_at=_parse_iso_datetime(row.get("created_at")),
    )
    if benchmark_return_pct is not None:
        detail["benchmark_return_pct"] = round(benchmark_return_pct, 4)
        detail["alpha_vs_benchmark_pct"] = round(float(return_pct) - benchmark_return_pct, 4)
        detail["alpha_after_cost_pct"] = round(return_after_cost_pct - benchmark_return_pct, 4)
        detail["benchmark_outperform_after_cost"] = bool((return_after_cost_pct - benchmark_return_pct) > 0)
    if benchmark_entry_price is not None:
        detail["benchmark_entry_price"] = round(benchmark_entry_price, 4)
    if benchmark_exit_price is not None:
        detail["benchmark_exit_price"] = round(benchmark_exit_price, 4)
    sector_benchmark_ticker = _resolve_sector_benchmark_ticker(detail=detail)
    peer_benchmark_ticker = str(detail.get("peer_benchmark_ticker") or sector_benchmark_ticker or "").strip().upper()
    detail["sector_benchmark_ticker"] = sector_benchmark_ticker
    detail["peer_benchmark_ticker"] = peer_benchmark_ticker
    sector_return_pct, _, _ = await _load_benchmark_return(
        services=services,
        benchmark_ticker=sector_benchmark_ticker,
        created_at=_parse_iso_datetime(row.get("created_at")),
    )
    if sector_return_pct is not None:
        detail["sector_benchmark_return_pct"] = round(sector_return_pct, 4)
        detail["alpha_vs_sector_pct"] = round(float(return_pct) - sector_return_pct, 4)
        detail["alpha_after_cost_vs_sector_pct"] = round(return_after_cost_pct - sector_return_pct, 4)
    peer_return_pct, _, _ = await _load_benchmark_return(
        services=services,
        benchmark_ticker=peer_benchmark_ticker,
        created_at=_parse_iso_datetime(row.get("created_at")),
    )
    if peer_return_pct is not None:
        detail["peer_benchmark_return_pct"] = round(peer_return_pct, 4)
        detail["alpha_after_cost_vs_peer_pct"] = round(return_after_cost_pct - peer_return_pct, 4)

    created_at = _parse_iso_datetime(row.get("created_at"))
    expires_at = _resolve_signal_expiry(created_at=created_at, detail=detail)
    stance = str(row.get("stance") or detail.get("stance") or "buy").strip().casefold() or "buy"
    detail["expired_before_evaluation"] = bool(expires_at is not None and expires_at <= now)
    if expires_at is not None:
        detail["signal_expires_at"] = expires_at.isoformat()

    ttl_reference = await _load_price_near_signal_expiry(
        services=services,
        ticker=ticker,
        created_at=created_at,
        expires_at=expires_at,
    )
    signed_final_return = _signed_signal_return(return_pct=return_pct, stance=stance)
    final_direction_hit = signed_final_return >= 0.02

    ttl_hit: bool | None = None
    ttl_return_pct: float | None = None
    if ttl_reference is not None:
        ttl_return_pct = (ttl_reference.close - entry_price) / entry_price
        ttl_hit = _signed_signal_return(return_pct=ttl_return_pct, stance=stance) >= 0.005
        detail["ttl_reference_price"] = round(ttl_reference.close, 4)
        detail["ttl_reference_at"] = ttl_reference.timestamp.isoformat()
        detail["ttl_return_pct"] = round(ttl_return_pct, 4)
        detail["ttl_return_after_cost_pct"] = round(ttl_return_pct - (execution_cost_bps / 10000.0), 4)
        detail["ttl_hit"] = ttl_hit

    decay_label, postmortem_action, postmortem_reason = _classify_signal_postmortem(
        ttl_hit=ttl_hit,
        final_direction_hit=final_direction_hit,
        return_pct=return_pct,
        stance=stance,
        expired_before_evaluation=bool(detail["expired_before_evaluation"]),
    )
    detail["signal_decay_label"] = decay_label
    detail["postmortem_action"] = postmortem_action
    detail["postmortem_reason"] = postmortem_reason
    return detail


def _classify_scorecard_outcome(*, return_pct: float) -> str:
    if return_pct >= 0.07:
        return "strong_win"
    if return_pct >= 0.02:
        return "win"
    if return_pct <= -0.07:
        return "strong_loss"
    if return_pct <= -0.02:
        return "loss"
    return "flat"


def _coerce_float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _hash_identifier(value: object) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        return ""
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def _resolve_benchmark_ticker(*, detail: dict[str, object]) -> str:
    explicit_ticker = str(detail.get("benchmark_ticker") or "").strip().upper()
    if explicit_ticker:
        return explicit_ticker
    benchmark = str(detail.get("benchmark") or "").strip().casefold()
    return {
        "sp500": "SPY",
        "sp500_index": "SPY",
        "nasdaq100": "QQQ",
        "nasdaq_index": "QQQ",
        "watchlist": "SPY",
        "custom": "SPY",
    }.get(benchmark, "SPY")


def _resolve_sector_benchmark_ticker(*, detail: dict[str, object]) -> str:
    sector = str(detail.get("sector") or "").strip().casefold()
    return {
        "technology": "XLK",
        "financials": "XLF",
        "energy": "XLE",
        "consumer discretionary": "XLY",
        "consumer staples": "XLP",
        "healthcare": "XLV",
        "industrials": "XLI",
        "materials": "XLB",
        "utilities": "XLU",
        "communication services": "XLC",
        "real estate": "XLRE",
    }.get(sector, _resolve_benchmark_ticker(detail=detail))


async def _load_benchmark_return(
    *,
    services: BotServices,
    benchmark_ticker: str,
    created_at: datetime | None,
) -> tuple[float | None, float | None, float | None]:
    if not benchmark_ticker:
        return None, None, None
    history_getter = getattr(services.market_data_client, "get_history", None)
    latest_getter = getattr(services.market_data_client, "get_latest_price", None)
    if not callable(history_getter):
        return None, None, None
    try:
        bars = await history_getter(
            benchmark_ticker,
            period=_select_history_period(created_at=created_at, expires_at=datetime.now(timezone.utc)),
            interval="1d",
            limit=180,
        )
    except Exception:
        bars = None
    if not isinstance(bars, list) or not bars:
        return None, None, None
    entry_bar = _find_bar_near_timestamp(
        bars=bars,
        target=created_at or datetime.now(timezone.utc),
        tolerance=timedelta(days=5),
    )
    latest_price = None
    if callable(latest_getter):
        try:
            latest_quote = await latest_getter(benchmark_ticker)
        except Exception:
            latest_quote = None
        latest_price = float(latest_quote.price) if latest_quote is not None and getattr(latest_quote, "price", None) else None
    if latest_price is None:
        latest_bar = max(
            bars,
            key=lambda item: item.timestamp if item.timestamp.tzinfo is not None else item.timestamp.replace(tzinfo=timezone.utc),
        )
        latest_price = float(latest_bar.close)
    if entry_bar is None or latest_price is None or float(entry_bar.close) <= 0:
        return None, float(entry_bar.close) if entry_bar is not None else None, latest_price
    return ((latest_price - float(entry_bar.close)) / float(entry_bar.close)), float(entry_bar.close), latest_price


async def _load_price_near_signal_expiry(
    *,
    services: BotServices,
    ticker: str,
    created_at: datetime | None,
    expires_at: datetime | None,
) -> OhlcvBar | None:
    if expires_at is None:
        return None
    history_getter = getattr(services.market_data_client, "get_history", None)
    if not callable(history_getter):
        return None
    try:
        bars = await history_getter(
            ticker,
            period=_select_history_period(created_at=created_at, expires_at=expires_at),
            interval="1h",
            limit=720,
        )
    except Exception:
        return None
    if not isinstance(bars, list) or not bars:
        return None
    return _find_bar_near_timestamp(bars=bars, target=expires_at, tolerance=timedelta(hours=8))


def _select_history_period(*, created_at: datetime | None, expires_at: datetime) -> str:
    anchor = created_at or expires_at
    span_days = max(1, math.ceil((datetime.now(timezone.utc) - anchor).total_seconds() / 86400.0) + 2)
    if span_days <= 5:
        return "5d"
    if span_days <= 30:
        return "1mo"
    if span_days <= 90:
        return "3mo"
    return "6mo"


def _find_bar_near_timestamp(
    *,
    bars: list[OhlcvBar],
    target: datetime,
    tolerance: timedelta,
) -> OhlcvBar | None:
    normalized_target = target if target.tzinfo is not None else target.replace(tzinfo=timezone.utc)
    matches: list[tuple[float, int, OhlcvBar]] = []
    for bar in bars:
        bar_ts = bar.timestamp if bar.timestamp.tzinfo is not None else bar.timestamp.replace(tzinfo=timezone.utc)
        distance = abs((bar_ts - normalized_target).total_seconds())
        if distance > tolerance.total_seconds():
            continue
        preference = 0 if bar_ts >= normalized_target else 1
        matches.append((distance, preference, bar))
    if not matches:
        return None
    matches.sort(key=lambda item: (item[0], item[1]))
    return matches[0][2]


def _resolve_signal_expiry(*, created_at: datetime | None, detail: dict[str, object]) -> datetime | None:
    explicit = _parse_iso_datetime(detail.get("expires_at"))
    if explicit is not None:
        return explicit
    if created_at is None:
        return None
    raw_ttl = detail.get("ttl_minutes")
    try:
        ttl_minutes = int(float(raw_ttl))
    except (TypeError, ValueError):
        return None
    return created_at + timedelta(minutes=max(1, ttl_minutes))


def _parse_iso_datetime(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _signed_signal_return(*, return_pct: float, stance: str) -> float:
    if stance in {"avoid", "sell", "reduce", "hedge"}:
        return -float(return_pct)
    return float(return_pct)


def _classify_signal_postmortem(
    *,
    ttl_hit: bool | None,
    final_direction_hit: bool,
    return_pct: float,
    stance: str,
    expired_before_evaluation: bool,
) -> tuple[str, str, str]:
    if not expired_before_evaluation:
        return "active", "hold_thesis", "signal is still active within its original shelf life"
    signed_final_return = _signed_signal_return(return_pct=return_pct, stance=stance)
    if ttl_hit is True and final_direction_hit:
        return "durable", "hold_thesis", "signal worked inside TTL and still held up by evaluation time"
    if ttl_hit is True and signed_final_return <= 0:
        return "fast_decay", "discard_thesis", "signal worked briefly but lost follow-through by evaluation time"
    if ttl_hit is False and final_direction_hit:
        return "late_follow_through", "hold_thesis", "signal missed the original TTL but the broader thesis kept working later"
    if ttl_hit is False and signed_final_return <= -0.02:
        return "expired_without_follow_through", "discard_thesis", "signal expired and the thesis failed to deliver follow-through"
    return "mixed", "revalidate_thesis", "signal expired with mixed follow-through and needs fresh confirmation"


def _chunk_text(text: str, *, limit: int) -> list[str]:
    normalized = text.strip()
    if not normalized:
        return []
    if len(normalized) <= limit:
        return [normalized]

    chunks: list[str] = []
    remaining = normalized
    while remaining:
        if len(remaining) <= limit:
            chunks.append(remaining)
            break
        split_at = remaining.rfind("\n", 0, limit)
        if split_at <= 0:
            split_at = remaining.rfind(" ", 0, limit)
        if split_at <= 0:
            split_at = limit
        chunks.append(remaining[:split_at].strip())
        remaining = remaining[split_at:].strip()
    return [chunk for chunk in chunks if chunk]


def _summarize_alert_categories(alerts: list[object]) -> str:
    categories = _summarize_alert_categories_map(alerts)
    return ",".join(f"{name}={count}" for name, count in sorted(categories.items()))


def _summarize_alert_categories_map(alerts: list[object]) -> dict[str, int]:
    categories: dict[str, int] = {}
    for alert in alerts:
        key = getattr(alert, "key", "")
        category = str(key).split(":", 1)[0] if str(key) else "unknown"
        categories[category] = categories.get(category, 0) + 1
    return categories


def _cleanup_old_logs(logs_dir: Path | None, retention_policy: str) -> int:
    if logs_dir is None or not logs_dir.exists():
        return 0
    retention_delta = _parse_retention_policy(retention_policy)
    if retention_delta is None:
        return 0
    deleted = 0
    from datetime import datetime, timezone as _timezone

    cutoff_dt = datetime.now(_timezone.utc) - retention_delta
    for pattern in ("*.log", "*.jsonl"):
        for path in logs_dir.glob(pattern):
            try:
                modified_at = datetime.fromtimestamp(path.stat().st_mtime, tz=_timezone.utc)
            except OSError:
                continue
            if modified_at >= cutoff_dt:
                continue
            try:
                path.unlink()
                deleted += 1
            except OSError:
                continue
    return deleted


def _parse_retention_policy(value: str) -> timedelta | None:
    normalized = value.strip().casefold()
    if not normalized:
        return None
    match = re.match(r"^(?P<number>\d+)\s*(?P<unit>day|days|hour|hours)$", normalized)
    if not match:
        return None
    amount = int(match.group("number"))
    unit = match.group("unit")
    if "hour" in unit:
        return timedelta(hours=amount)
    return timedelta(days=amount)
