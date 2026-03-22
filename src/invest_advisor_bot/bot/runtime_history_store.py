from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Mapping

from invest_advisor_bot.bot.postgres_state import PostgresStateBackend


class RuntimeHistoryStore:
    """Persist runtime/audit history for diagnostics, reports, alerts, and user interactions."""

    def __init__(self, *, path: Path, database_url: str = "", retention_days: int = 30) -> None:
        self.path = path
        self.retention_days = max(7, int(retention_days))
        self._lock = RLock()
        self._db = PostgresStateBackend(database_url=database_url) if database_url.strip() else None
        if self._db is not None:
            self._db.ensure_schema()

    def record_provider_event(
        self,
        *,
        provider: str,
        model: str | None,
        service: str | None,
        status: str,
        detail: Mapping[str, Any] | None = None,
    ) -> None:
        if self._db is None:
            return
        self._db.execute(
            """
            INSERT INTO bot_provider_diagnostics_history (provider, model, service, status, detail)
            VALUES (%s, %s, %s, %s, %s::jsonb)
            """,
            (provider, model, service, status, json.dumps(dict(detail or {}), ensure_ascii=False)),
        )

    def record_job_event(
        self,
        *,
        job_name: str,
        status: str,
        duration_ms: int | None,
        detail: Mapping[str, Any] | None = None,
    ) -> None:
        if self._db is None:
            return
        self._db.execute(
            """
            INSERT INTO bot_job_diagnostics_history (job_name, status, duration_ms, detail)
            VALUES (%s, %s, %s, %s::jsonb)
            """,
            (job_name, status, duration_ms, json.dumps(dict(detail or {}), ensure_ascii=False)),
        )

    def record_sent_report(
        self,
        *,
        report_kind: str,
        chat_id: str,
        fallback_used: bool,
        model: str | None,
        summary: str,
        detail: Mapping[str, Any] | None = None,
    ) -> None:
        if self._db is None:
            return
        self._db.execute(
            """
            INSERT INTO bot_sent_reports_history (report_kind, chat_id, fallback_used, model, summary, detail)
            VALUES (%s, %s, %s, %s, %s, %s::jsonb)
            """,
            (report_kind, chat_id, fallback_used, model, summary, json.dumps(dict(detail or {}), ensure_ascii=False)),
        )

    def record_user_interaction(
        self,
        *,
        conversation_key: str | None,
        interaction_kind: str,
        question: str | None,
        response_excerpt: str,
        fallback_used: bool,
        model: str | None,
        detail: Mapping[str, Any] | None = None,
    ) -> None:
        if self._db is None:
            return
        self._db.execute(
            """
            INSERT INTO bot_user_interaction_history (
                conversation_key, interaction_kind, question, response_excerpt, fallback_used, model, detail
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb)
            """,
            (
                conversation_key,
                interaction_kind,
                question,
                response_excerpt,
                fallback_used,
                model,
                json.dumps(dict(detail or {}), ensure_ascii=False),
            ),
        )

    def record_alert_audit(
        self,
        *,
        alert_key: str,
        severity: str,
        category: str,
        chat_id: str,
        text_excerpt: str,
        detail: Mapping[str, Any] | None = None,
    ) -> None:
        if self._db is None:
            return
        self._db.execute(
            """
            INSERT INTO bot_alert_audit (alert_key, severity, category, chat_id, text_excerpt, detail)
            VALUES (%s, %s, %s, %s, %s, %s::jsonb)
            """,
            (
                alert_key,
                severity,
                category,
                chat_id,
                text_excerpt,
                json.dumps(dict(detail or {}), ensure_ascii=False),
            ),
        )

    def record_stock_pick_candidate(
        self,
        *,
        recommendation_key: str,
        due_at: datetime,
        source_kind: str,
        chat_id: str | None,
        ticker: str,
        company_name: str | None,
        stance: str | None,
        confidence_score: float | None,
        confidence_label: str | None,
        composite_score: float | None,
        entry_price: float,
        detail: Mapping[str, Any] | None = None,
    ) -> None:
        if self._db is None:
            return
        self._db.execute(
            """
            INSERT INTO bot_stock_pick_scorecard (
                recommendation_key, due_at, source_kind, chat_id, ticker, company_name, stance,
                confidence_score, confidence_label, composite_score, entry_price, detail
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
            ON CONFLICT (recommendation_key) DO NOTHING
            """,
            (
                recommendation_key,
                due_at,
                source_kind,
                chat_id,
                ticker,
                company_name,
                stance,
                confidence_score,
                confidence_label,
                composite_score,
                entry_price,
                json.dumps(dict(detail or {}), ensure_ascii=False),
            ),
        )

    def list_due_stock_pick_candidates(self, *, limit: int = 20) -> list[dict[str, Any]]:
        if self._db is None:
            return []
        rows = self._db.fetch_all(
            """
            SELECT id, recommendation_key, due_at, source_kind, chat_id, ticker, company_name, stance,
                   confidence_score, confidence_label, composite_score, entry_price, detail
            FROM bot_stock_pick_scorecard
            WHERE status = 'open' AND due_at <= NOW()
            ORDER BY due_at ASC
            LIMIT %s
            """,
            (max(1, limit),),
        )
        return [
            {
                "id": row[0],
                "recommendation_key": row[1],
                "due_at": row[2].isoformat() if isinstance(row[2], datetime) else None,
                "source_kind": row[3],
                "chat_id": row[4],
                "ticker": row[5],
                "company_name": row[6],
                "stance": row[7],
                "confidence_score": row[8],
                "confidence_label": row[9],
                "composite_score": row[10],
                "entry_price": row[11],
                "detail": row[12],
            }
            for row in rows
        ]

    def complete_stock_pick_evaluation(
        self,
        *,
        record_id: int,
        exit_price: float,
        return_pct: float,
        outcome_label: str,
        detail: Mapping[str, Any] | None = None,
    ) -> None:
        if self._db is None:
            return
        self._db.execute(
            """
            UPDATE bot_stock_pick_scorecard
            SET evaluated_at = NOW(),
                exit_price = %s,
                return_pct = %s,
                outcome_label = %s,
                status = 'closed',
                detail = COALESCE(detail, '{}'::jsonb) || %s::jsonb
            WHERE id = %s
            """,
            (
                exit_price,
                return_pct,
                outcome_label,
                json.dumps(dict(detail or {}), ensure_ascii=False),
                int(record_id),
            ),
        )

    def recent_stock_pick_scorecard(self, *, chat_id: str | None = None, limit: int = 10) -> list[dict[str, Any]]:
        if self._db is None:
            return []
        if chat_id:
            rows = self._db.fetch_all(
                """
                SELECT recommendation_key, created_at, source_kind, ticker, company_name, stance,
                       confidence_score, confidence_label, composite_score, entry_price,
                       evaluated_at, exit_price, return_pct, outcome_label, status, detail
                FROM bot_stock_pick_scorecard
                WHERE chat_id = %s
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (chat_id, max(1, limit)),
            )
        else:
            rows = self._db.fetch_all(
                """
                SELECT recommendation_key, created_at, source_kind, ticker, company_name, stance,
                       confidence_score, confidence_label, composite_score, entry_price,
                       evaluated_at, exit_price, return_pct, outcome_label, status, detail
                FROM bot_stock_pick_scorecard
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (max(1, limit),),
            )
        return [
            {
                "recommendation_key": row[0],
                "created_at": row[1].isoformat() if isinstance(row[1], datetime) else None,
                "source_kind": row[2],
                "ticker": row[3],
                "company_name": row[4],
                "stance": row[5],
                "confidence_score": row[6],
                "confidence_label": row[7],
                "composite_score": row[8],
                "entry_price": row[9],
                "evaluated_at": row[10].isoformat() if isinstance(row[10], datetime) else None,
                "exit_price": row[11],
                "return_pct": row[12],
                "outcome_label": row[13],
                "status": row[14],
                "detail": row[15],
            }
            for row in rows
        ]

    def recent_job_events(self, *, limit: int = 5) -> list[dict[str, Any]]:
        if self._db is None:
            return []
        rows = self._db.fetch_all(
            """
            SELECT job_name, status, duration_ms, event_at, detail
            FROM bot_job_diagnostics_history
            ORDER BY event_at DESC
            LIMIT %s
            """,
            (max(1, limit),),
        )
        return [
            {
                "job_name": row[0],
                "status": row[1],
                "duration_ms": row[2],
                "event_at": row[3].isoformat() if isinstance(row[3], datetime) else None,
                "detail": row[4],
            }
            for row in rows
        ]

    def build_evaluation_dashboard(
        self,
        *,
        chat_id: str | None = None,
        lookback_days: int = 14,
        burn_in_target_days: int = 14,
    ) -> dict[str, Any]:
        if self._db is None:
            return {
                "available": False,
                "burn_in": {
                    "started_at": None,
                    "elapsed_days": 0.0,
                    "target_days": max(7, int(burn_in_target_days)),
                    "progress_pct": 0.0,
                },
                "providers": [],
                "jobs": [],
                "alerts": {"total": 0, "by_category": []},
                "reports": {"total": 0, "fallback_total": 0, "by_kind": []},
                "scorecard": {"closed_count": 0, "open_count": 0, "hit_rate_pct": 0.0, "avg_return_pct": None},
                "interactions": {"total": 0, "last_at": None},
            }

        effective_lookback = max(1, int(lookback_days))
        target_days = max(7, int(burn_in_target_days))
        started_at_row = self._db.fetch_one(
            """
            SELECT MIN(ts)
            FROM (
                SELECT MIN(event_at) AS ts FROM bot_provider_diagnostics_history
                UNION ALL
                SELECT MIN(event_at) AS ts FROM bot_job_diagnostics_history
                UNION ALL
                SELECT MIN(sent_at) AS ts FROM bot_sent_reports_history
                UNION ALL
                SELECT MIN(occurred_at) AS ts FROM bot_user_interaction_history
                UNION ALL
                SELECT MIN(sent_at) AS ts FROM bot_alert_audit
                UNION ALL
                SELECT MIN(created_at) AS ts FROM bot_stock_pick_scorecard
            ) AS burn_in
            WHERE ts IS NOT NULL
            """,
        )
        started_at = started_at_row[0] if started_at_row is not None and isinstance(started_at_row[0], datetime) else None
        elapsed_days = 0.0
        if started_at is not None:
            elapsed_days = round((datetime.now(timezone.utc) - started_at).total_seconds() / 86400.0, 2)

        providers = self._db.fetch_all(
            """
            SELECT provider,
                   COUNT(*) FILTER (WHERE status = 'success') AS success_count,
                   COUNT(*) FILTER (WHERE status = 'failure') AS failure_count,
                   MAX(event_at) AS last_event_at
            FROM bot_provider_diagnostics_history
            WHERE event_at >= NOW() - (%s * INTERVAL '1 day')
            GROUP BY provider
            ORDER BY failure_count DESC, success_count DESC, provider ASC
            LIMIT 12
            """,
            (effective_lookback,),
        )
        jobs = self._db.fetch_all(
            """
            SELECT job_name,
                   COUNT(*) FILTER (WHERE status = 'ok') AS success_count,
                   COUNT(*) FILTER (WHERE status <> 'ok') AS failure_count,
                   MAX(event_at) AS last_event_at
            FROM bot_job_diagnostics_history
            WHERE event_at >= NOW() - (%s * INTERVAL '1 day')
            GROUP BY job_name
            ORDER BY failure_count DESC, success_count DESC, job_name ASC
            LIMIT 12
            """,
            (effective_lookback,),
        )
        alerts = self._db.fetch_all(
            """
            SELECT category, COUNT(*) AS total
            FROM bot_alert_audit
            WHERE sent_at >= NOW() - (%s * INTERVAL '1 day')
            GROUP BY category
            ORDER BY total DESC, category ASC
            LIMIT 12
            """,
            (effective_lookback,),
        )
        reports_where = "WHERE sent_at >= NOW() - (%s * INTERVAL '1 day')"
        reports_params: list[Any] = [effective_lookback]
        if chat_id:
            reports_where += " AND chat_id = %s"
            reports_params.append(chat_id)
        reports_summary_row = self._db.fetch_one(
            f"""
            SELECT COUNT(*) AS total,
                   COUNT(*) FILTER (WHERE fallback_used) AS fallback_total,
                   MAX(sent_at) AS last_sent_at
            FROM bot_sent_reports_history
            {reports_where}
            """,
            tuple(reports_params),
        )
        reports_kind = self._db.fetch_all(
            f"""
            SELECT report_kind, COUNT(*) AS total
            FROM bot_sent_reports_history
            {reports_where}
            GROUP BY report_kind
            ORDER BY total DESC, report_kind ASC
            LIMIT 10
            """,
            tuple(reports_params),
        )

        scorecard_where = "WHERE created_at >= NOW() - (%s * INTERVAL '1 day')"
        scorecard_params: list[Any] = [effective_lookback]
        if chat_id:
            scorecard_where += " AND chat_id = %s"
            scorecard_params.append(chat_id)
        scorecard_row = self._db.fetch_one(
            f"""
            SELECT COUNT(*) FILTER (WHERE status = 'closed') AS closed_count,
                   COUNT(*) FILTER (WHERE status = 'open') AS open_count,
                   COUNT(*) FILTER (WHERE status = 'closed' AND return_pct > 0) AS win_count,
                   COUNT(*) FILTER (WHERE status = 'closed' AND return_pct >= 0.07) AS strong_win_count,
                   COUNT(*) FILTER (WHERE status = 'closed' AND return_pct <= -0.07) AS strong_loss_count,
                   AVG(return_pct) FILTER (WHERE status = 'closed') AS avg_return_pct,
                   MAX(evaluated_at) AS last_evaluated_at
            FROM bot_stock_pick_scorecard
            {scorecard_where}
            """,
            tuple(scorecard_params),
        )

        interaction_where = "WHERE occurred_at >= NOW() - (%s * INTERVAL '1 day')"
        interaction_params: list[Any] = [effective_lookback]
        if chat_id:
            interaction_where += " AND conversation_key = %s"
            interaction_params.append(chat_id)
        interaction_row = self._db.fetch_one(
            f"""
            SELECT COUNT(*) AS total, MAX(occurred_at) AS last_at
            FROM bot_user_interaction_history
            {interaction_where}
            """,
            tuple(interaction_params),
        )

        closed_count = int(scorecard_row[0] or 0) if scorecard_row else 0
        win_count = int(scorecard_row[2] or 0) if scorecard_row else 0
        hit_rate_pct = round((win_count / closed_count) * 100.0, 1) if closed_count else 0.0
        avg_return_pct = float(scorecard_row[5]) * 100.0 if scorecard_row and scorecard_row[5] is not None else None

        return {
            "available": True,
            "burn_in": {
                "started_at": started_at.isoformat() if started_at is not None else None,
                "elapsed_days": elapsed_days,
                "target_days": target_days,
                "progress_pct": round(min(100.0, (elapsed_days / target_days) * 100.0), 1) if elapsed_days else 0.0,
            },
            "providers": [
                {
                    "provider": row[0],
                    "success_count": int(row[1] or 0),
                    "failure_count": int(row[2] or 0),
                    "last_event_at": row[3].isoformat() if isinstance(row[3], datetime) else None,
                }
                for row in providers
            ],
            "jobs": [
                {
                    "job_name": row[0],
                    "success_count": int(row[1] or 0),
                    "failure_count": int(row[2] or 0),
                    "last_event_at": row[3].isoformat() if isinstance(row[3], datetime) else None,
                }
                for row in jobs
            ],
            "alerts": {
                "total": sum(int(row[1] or 0) for row in alerts),
                "by_category": [{"category": row[0], "total": int(row[1] or 0)} for row in alerts],
            },
            "reports": {
                "total": int(reports_summary_row[0] or 0) if reports_summary_row else 0,
                "fallback_total": int(reports_summary_row[1] or 0) if reports_summary_row else 0,
                "last_sent_at": (
                    reports_summary_row[2].isoformat()
                    if reports_summary_row and isinstance(reports_summary_row[2], datetime)
                    else None
                ),
                "by_kind": [{"report_kind": row[0], "total": int(row[1] or 0)} for row in reports_kind],
            },
            "scorecard": {
                "closed_count": closed_count,
                "open_count": int(scorecard_row[1] or 0) if scorecard_row else 0,
                "win_count": win_count,
                "strong_win_count": int(scorecard_row[3] or 0) if scorecard_row else 0,
                "strong_loss_count": int(scorecard_row[4] or 0) if scorecard_row else 0,
                "hit_rate_pct": hit_rate_pct,
                "avg_return_pct": round(avg_return_pct, 2) if avg_return_pct is not None else None,
                "last_evaluated_at": (
                    scorecard_row[6].isoformat() if scorecard_row and isinstance(scorecard_row[6], datetime) else None
                ),
            },
            "interactions": {
                "total": int(interaction_row[0] or 0) if interaction_row else 0,
                "last_at": interaction_row[1].isoformat() if interaction_row and isinstance(interaction_row[1], datetime) else None,
            },
        }

    def cleanup_retention(self) -> None:
        if self._db is None:
            return
        self._cleanup_table("bot_provider_diagnostics_history", "event_at")
        self._cleanup_table("bot_job_diagnostics_history", "event_at")
        self._cleanup_table("bot_sent_reports_history", "sent_at")
        self._cleanup_table("bot_user_interaction_history", "occurred_at")
        self._cleanup_table("bot_alert_audit", "sent_at")
        self._cleanup_table("bot_stock_pick_scorecard", "created_at")

    def _cleanup_table(self, table_name: str, time_column: str) -> None:
        if self._db is None:
            return
        self._db.execute(
            f"DELETE FROM {table_name} WHERE {time_column} < NOW() - (%s * INTERVAL '1 day')",
            (self.retention_days,),
        )
