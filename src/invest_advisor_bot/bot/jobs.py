from __future__ import annotations

import asyncio
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from time import perf_counter
import re

from loguru import logger
from telegram.ext import Application, ContextTypes

from invest_advisor_bot.bot.handlers import BOT_SERVICES_KEY, BotServices
from invest_advisor_bot.observability import log_event
from invest_advisor_bot.runtime_diagnostics import diagnostics


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
        for chunk in _chunk_text(result.recommendation_text, limit=3900):
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
        if services.alert_state_store is not None:
            allowed_keys = set(services.alert_state_store.filter_new_keys(alert.key for alert in alerts))
            alerts = [alert for alert in alerts if alert.key in allowed_keys]
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
    watchlist = prefs.watchlist if prefs is not None else ()

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
        )
        if services.alert_state_store is not None:
            allowed_keys = set(services.alert_state_store.filter_new_keys(alert.key for alert in alerts))
            alerts = [alert for alert in alerts if alert.key in allowed_keys]
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
        if services.alert_state_store is not None:
            allowed_keys = set(services.alert_state_store.filter_new_keys(alert.key for alert in alerts))
            alerts = [alert for alert in alerts if alert.key in allowed_keys]
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
            if return_pct >= 0.07:
                outcome_label = "strong_win"
            elif return_pct >= 0.02:
                outcome_label = "win"
            elif return_pct <= -0.07:
                outcome_label = "strong_loss"
            elif return_pct <= -0.02:
                outcome_label = "loss"
            else:
                outcome_label = "flat"
            services.runtime_history_store.complete_stock_pick_evaluation(
                record_id=int(record_id),
                exit_price=float(quote.price),
                return_pct=float(return_pct),
                outcome_label=outcome_label,
                detail={"evaluated_at": datetime.now(timezone.utc).isoformat()},
            )
            evaluated_count += 1
            summary_lines.append(
                f"- {ticker}: {return_pct:+.1%} | outcome {outcome_label} | source {row.get('source_kind') or '-'}"
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
    earnings_alert_days_ahead: int,
    maintenance_cleanup_interval_minutes: int,
    stock_pick_evaluation_interval_minutes: int,
    backup_interval_hours: int,
) -> None:
    jq = application.job_queue
    if jq is None:
        logger.warning("Job queue is not available; scheduled jobs will be skipped")
        return

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
        for chunk in _chunk_text(result.recommendation_text, limit=3900):
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


def _record_sent_report(services: BotServices, report_kind: str, chat_id: str, result: object) -> None:
    if services.runtime_history_store is None:
        return
    recommendation_text = str(getattr(result, "recommendation_text", "") or "")
    fallback_used = bool(getattr(result, "fallback_used", False))
    model = getattr(result, "model", None)
    services.runtime_history_store.record_sent_report(
        report_kind=report_kind,
        chat_id=str(chat_id),
        fallback_used=fallback_used,
        model=str(model) if isinstance(model, str) else None,
        summary=recommendation_text[:240],
        detail={"source": "scheduled_job"},
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
            detail={},
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
            detail={"due_at": due_at.isoformat()},
        )


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
