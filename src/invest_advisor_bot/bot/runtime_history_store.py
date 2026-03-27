from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Mapping

from invest_advisor_bot.bot.postgres_state import PostgresStateBackend


class RuntimeHistoryStore:
    """Persist runtime/audit history for diagnostics, reports, alerts, and user interactions."""

    def __init__(self, *, path: Path, database_url: str = "", retention_days: int = 30, pgvector_enabled: bool = True) -> None:
        self.path = path
        self.retention_days = max(7, int(retention_days))
        self.pgvector_enabled = bool(pgvector_enabled)
        self._lock = RLock()
        self._db = PostgresStateBackend(database_url=database_url) if database_url.strip() else None
        if self._db is not None:
            self._db.ensure_schema()

    def status(self) -> dict[str, Any]:
        pgvector_ready = self._db.vector_ready() if self._db is not None and self.pgvector_enabled else False
        return {
            "available": True,
            "backend": "postgres" if self._db is not None else "file",
            "path": str(self.path),
            "retention_days": self.retention_days,
            "pgvector_enabled": self.pgvector_enabled,
            "pgvector_ready": pgvector_ready,
        }

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

    def record_thesis_memory(
        self,
        *,
        thesis_key: str,
        thesis_text: str,
        source_kind: str,
        conversation_key: str | None = None,
        query_text: str | None = None,
        tags: list[str] | tuple[str, ...] | None = None,
        confidence_score: float | None = None,
        detail: Mapping[str, Any] | None = None,
    ) -> None:
        if self._db is None:
            return
        normalized_text = (thesis_text or "").strip()
        if not normalized_text:
            return
        embedding = self._embed_text(normalized_text)
        tags_payload = [str(item).strip() for item in (tags or []) if str(item).strip()]
        detail_payload = json.dumps(dict(detail or {}), ensure_ascii=False)
        if self.pgvector_enabled and self._db.vector_ready():
            self._db.execute(
                """
                INSERT INTO bot_thesis_memory (
                    conversation_key, thesis_key, query_text, thesis_text, source_kind,
                    tags, confidence_score, embedding_json, embedding, detail
                )
                VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s, %s::jsonb, %s::vector, %s::jsonb)
                ON CONFLICT (thesis_key) DO UPDATE
                SET query_text = EXCLUDED.query_text,
                    thesis_text = EXCLUDED.thesis_text,
                    source_kind = EXCLUDED.source_kind,
                    tags = EXCLUDED.tags,
                    confidence_score = EXCLUDED.confidence_score,
                    embedding_json = EXCLUDED.embedding_json,
                    embedding = EXCLUDED.embedding,
                    detail = COALESCE(bot_thesis_memory.detail, '{}'::jsonb) || EXCLUDED.detail
                """,
                (
                    conversation_key,
                    thesis_key,
                    query_text,
                    normalized_text,
                    source_kind,
                    json.dumps(tags_payload, ensure_ascii=False),
                    confidence_score,
                    json.dumps(embedding, ensure_ascii=False),
                    self._vector_literal(embedding),
                    detail_payload,
                ),
            )
            return
        self._db.execute(
            """
            INSERT INTO bot_thesis_memory (
                conversation_key, thesis_key, query_text, thesis_text, source_kind,
                tags, confidence_score, embedding_json, detail
            )
            VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s, %s::jsonb, %s::jsonb)
            ON CONFLICT (thesis_key) DO UPDATE
            SET query_text = EXCLUDED.query_text,
                thesis_text = EXCLUDED.thesis_text,
                source_kind = EXCLUDED.source_kind,
                tags = EXCLUDED.tags,
                confidence_score = EXCLUDED.confidence_score,
                embedding_json = EXCLUDED.embedding_json,
                detail = COALESCE(bot_thesis_memory.detail, '{}'::jsonb) || EXCLUDED.detail
            """,
            (
                conversation_key,
                thesis_key,
                query_text,
                normalized_text,
                source_kind,
                json.dumps(tags_payload, ensure_ascii=False),
                confidence_score,
                json.dumps(embedding, ensure_ascii=False),
                detail_payload,
            ),
        )

    def search_thesis_memory(
        self,
        *,
        query_text: str,
        conversation_key: str | None = None,
        limit: int = 3,
    ) -> list[dict[str, Any]]:
        if self._db is None:
            return []
        normalized_query = (query_text or "").strip()
        if not normalized_query:
            return []
        effective_limit = max(1, int(limit))
        if self.pgvector_enabled and self._db.vector_ready():
            embedding = self._embed_text(normalized_query)
            rows = self._db.fetch_dicts(
                """
                SELECT thesis_key, conversation_key, thesis_text, source_kind, query_text,
                       tags, confidence_score, detail, created_at,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM bot_thesis_memory
                WHERE (%s IS NULL OR conversation_key = %s OR conversation_key IS NULL)
                ORDER BY embedding <=> %s::vector ASC, created_at DESC
                LIMIT %s
                """,
                (
                    self._vector_literal(embedding),
                    conversation_key,
                    conversation_key,
                    self._vector_literal(embedding),
                    effective_limit,
                ),
            )
        else:
            tokens = [token for token in re.findall(r"[A-Za-z0-9_]+", normalized_query.casefold()) if len(token) >= 3]
            conditions = " OR ".join("thesis_text ILIKE %s" for _ in tokens[:4]) or "TRUE"
            params: list[Any] = [f"%{token}%" for token in tokens[:4]]
            params.extend([conversation_key, conversation_key, effective_limit])
            rows = self._db.fetch_dicts(
                f"""
                SELECT thesis_key, conversation_key, thesis_text, source_kind, query_text,
                       tags, confidence_score, detail, created_at, NULL::double precision AS similarity
                FROM bot_thesis_memory
                WHERE ({conditions})
                  AND (%s IS NULL OR conversation_key = %s OR conversation_key IS NULL)
                ORDER BY created_at DESC
                LIMIT %s
                """,
                tuple(params),
            )
        results: list[dict[str, Any]] = []
        for row in rows:
            created_at = row.get("created_at")
            results.append(
                {
                    "thesis_key": row.get("thesis_key"),
                    "conversation_key": row.get("conversation_key"),
                    "thesis_text": row.get("thesis_text"),
                    "source_kind": row.get("source_kind"),
                    "query_text": row.get("query_text"),
                    "tags": row.get("tags") or [],
                    "confidence_score": row.get("confidence_score"),
                    "detail": row.get("detail") or {},
                    "created_at": created_at.isoformat() if isinstance(created_at, datetime) else None,
                    "similarity": row.get("similarity"),
                }
            )
        return results

    def record_evaluation_artifact(
        self,
        *,
        artifact_key: str,
        artifact_kind: str,
        conversation_key: str | None = None,
        model: str | None = None,
        fallback_used: bool = False,
        metrics: Mapping[str, Any] | None = None,
        detail: Mapping[str, Any] | None = None,
    ) -> None:
        if self._db is None:
            return
        self._db.execute(
            """
            INSERT INTO bot_eval_artifact (
                artifact_key, artifact_kind, conversation_key, model, fallback_used, metrics, detail
            )
            VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s::jsonb)
            ON CONFLICT (artifact_key) DO UPDATE
            SET metrics = EXCLUDED.metrics,
                detail = COALESCE(bot_eval_artifact.detail, '{}'::jsonb) || EXCLUDED.detail,
                model = EXCLUDED.model,
                fallback_used = EXCLUDED.fallback_used
            """,
            (
                artifact_key,
                artifact_kind,
                conversation_key,
                model,
                fallback_used,
                json.dumps(dict(metrics or {}), ensure_ascii=False),
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
                   confidence_score, confidence_label, composite_score, entry_price, detail, created_at
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
                "created_at": row[13].isoformat() if isinstance(row[13], datetime) else None,
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
        execution_alert_kind: str | None = None,
    ) -> dict[str, Any]:
        normalized_execution_filter = self._normalize_execution_alert_kind(execution_alert_kind)
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
                "scorecard": {
                    "closed_count": 0,
                    "open_count": 0,
                    "hit_rate_pct": 0.0,
                    "avg_return_pct": None,
                    "avg_return_after_cost_pct": None,
                    "avg_alpha_pct": None,
                    "avg_alpha_after_cost_pct": None,
                },
                "interactions": {"total": 0, "last_at": None},
                "source_ranking": [],
                "thesis_ranking": [],
                "decision_quality": {
                    "source_health": {
                        "sample_count": 0,
                        "avg_score": None,
                        "avg_freshness_pct": None,
                        "degraded_sla_count": 0,
                        "outage_count": 0,
                        "strong_count": 0,
                        "mixed_count": 0,
                        "fragile_count": 0,
                    },
                    "no_trade": {
                        "decision_count": 0,
                        "abstain_count": 0,
                        "abstain_rate_pct": None,
                        "top_reasons": [],
                    },
                },
                "execution_panel": {
                    "alert_kind_filter": normalized_execution_filter,
                    "closed_postmortems": 0,
                    "ttl_hit_rate_pct": None,
                    "fast_decay_rate_pct": None,
                    "hold_after_expiry_rate_pct": None,
                    "discard_after_expiry_rate_pct": None,
                    "by_alert_kind": [],
                    "best_ttl_by_alert_kind": [],
                    "source_ttl_heatmap": [],
                },
                "thesis_lifecycle": {"counts": [], "top_invalidations": []},
                "walk_forward_eval": {"window_size": 5, "window_count": 0, "avg_hit_rate_pct": None, "avg_return_after_cost_pct": None},
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
        report_detail_rows = self._db.fetch_all(
            f"""
            SELECT detail
            FROM bot_sent_reports_history
            {reports_where}
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
        scorecard_detail_rows = self._db.fetch_all(
            f"""
            SELECT detail, return_pct, status, created_at
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
        interaction_detail_rows = self._db.fetch_all(
            f"""
            SELECT detail
            FROM bot_user_interaction_history
            {interaction_where}
            """,
            tuple(interaction_params),
        )

        closed_count = int(scorecard_row[0] or 0) if scorecard_row else 0
        win_count = int(scorecard_row[2] or 0) if scorecard_row else 0
        hit_rate_pct = round((win_count / closed_count) * 100.0, 1) if closed_count else 0.0
        avg_return_pct = float(scorecard_row[5]) * 100.0 if scorecard_row and scorecard_row[5] is not None else None
        avg_return_after_cost_pct = self._compute_avg_return_after_cost_pct(scorecard_detail_rows=scorecard_detail_rows)
        avg_alpha_pct = self._compute_avg_scorecard_detail_pct(
            scorecard_detail_rows=scorecard_detail_rows,
            field_name="alpha_vs_benchmark_pct",
        )
        avg_alpha_after_cost_pct = self._compute_avg_scorecard_detail_pct(
            scorecard_detail_rows=scorecard_detail_rows,
            field_name="alpha_after_cost_pct",
        )
        thesis_stats = self._build_thesis_stats(scorecard_detail_rows=scorecard_detail_rows)
        source_ranking = self._build_source_ranking(
            report_detail_rows=report_detail_rows,
            interaction_detail_rows=interaction_detail_rows,
            scorecard_detail_rows=scorecard_detail_rows,
            thesis_stats=thesis_stats,
        )
        thesis_ranking = self._serialize_thesis_ranking(thesis_stats=thesis_stats)
        decision_quality = self._build_decision_quality_snapshot(
            report_detail_rows=report_detail_rows,
            interaction_detail_rows=interaction_detail_rows,
            scorecard_detail_rows=scorecard_detail_rows,
        )
        thesis_lifecycle = self._build_thesis_lifecycle_snapshot(
            report_detail_rows=report_detail_rows,
            interaction_detail_rows=interaction_detail_rows,
            scorecard_detail_rows=scorecard_detail_rows,
        )
        walk_forward_eval = self._build_walk_forward_eval(scorecard_detail_rows=scorecard_detail_rows)
        execution_panel = self._build_execution_panel(
            scorecard_detail_rows=scorecard_detail_rows,
            alert_kind_filter=normalized_execution_filter,
        )

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
                "avg_return_after_cost_pct": avg_return_after_cost_pct,
                "avg_alpha_pct": avg_alpha_pct,
                "avg_alpha_after_cost_pct": avg_alpha_after_cost_pct,
                "last_evaluated_at": (
                    scorecard_row[6].isoformat() if scorecard_row and isinstance(scorecard_row[6], datetime) else None
                ),
            },
            "interactions": {
                "total": int(interaction_row[0] or 0) if interaction_row else 0,
                "last_at": interaction_row[1].isoformat() if interaction_row and isinstance(interaction_row[1], datetime) else None,
            },
            "source_ranking": source_ranking,
            "thesis_ranking": thesis_ranking,
            "decision_quality": decision_quality,
            "thesis_lifecycle": thesis_lifecycle,
            "walk_forward_eval": walk_forward_eval,
            "execution_panel": execution_panel,
        }

    def _build_decision_quality_snapshot(
        self,
        *,
        report_detail_rows: list[Any],
        interaction_detail_rows: list[Any],
        scorecard_detail_rows: list[Any],
    ) -> dict[str, Any]:
        health_sample_count = 0
        health_score_sum = 0.0
        health_freshness_sum = 0.0
        health_freshness_count = 0
        degraded_sla_count = 0
        outage_count = 0
        strong_count = 0
        mixed_count = 0
        fragile_count = 0
        decision_count = 0
        abstain_count = 0
        reason_counts: dict[str, int] = {}

        for detail in self._iter_dashboard_details(
            report_detail_rows=report_detail_rows,
            interaction_detail_rows=interaction_detail_rows,
            scorecard_detail_rows=scorecard_detail_rows,
        ):
            source_health = detail.get("source_health")
            if isinstance(source_health, Mapping):
                try:
                    score = float(source_health.get("score"))
                except (TypeError, ValueError):
                    score = None
                try:
                    freshness_pct = float(source_health.get("freshness_pct"))
                except (TypeError, ValueError):
                    freshness_pct = None
                if score is not None:
                    health_sample_count += 1
                    health_score_sum += score
                    if score >= 72.0:
                        strong_count += 1
                    elif score >= 52.0:
                        mixed_count += 1
                    else:
                        fragile_count += 1
                if freshness_pct is not None:
                    health_freshness_sum += freshness_pct
                    health_freshness_count += 1
                if str(source_health.get("sla_status") or "").strip().casefold() == "degraded":
                    degraded_sla_count += 1
                if bool(source_health.get("outage_detected")):
                    outage_count += 1

            no_trade = detail.get("no_trade_decision")
            if not isinstance(no_trade, Mapping):
                continue
            decision_count += 1
            if bool(no_trade.get("should_abstain")):
                abstain_count += 1
            reasons = no_trade.get("reasons")
            if isinstance(reasons, list):
                for reason in reasons[:4]:
                    reason_text = str(reason or "").strip()
                    if not reason_text:
                        continue
                    reason_counts[reason_text] = reason_counts.get(reason_text, 0) + 1

        top_reasons = [
            {"reason": reason, "count": count}
            for reason, count in sorted(reason_counts.items(), key=lambda item: (-item[1], item[0]))[:5]
        ]
        return {
            "source_health": {
                "sample_count": health_sample_count,
                "avg_score": round(health_score_sum / health_sample_count, 1) if health_sample_count else None,
                "avg_freshness_pct": (
                    round(health_freshness_sum / health_freshness_count, 1) if health_freshness_count else None
                ),
                "degraded_sla_count": degraded_sla_count,
                "outage_count": outage_count,
                "strong_count": strong_count,
                "mixed_count": mixed_count,
                "fragile_count": fragile_count,
            },
            "no_trade": {
                "decision_count": decision_count,
                "abstain_count": abstain_count,
                "abstain_rate_pct": round((abstain_count / decision_count) * 100.0, 1) if decision_count else None,
                "top_reasons": top_reasons,
            },
        }

    def _build_thesis_lifecycle_snapshot(
        self,
        *,
        report_detail_rows: list[Any],
        interaction_detail_rows: list[Any],
        scorecard_detail_rows: list[Any],
    ) -> dict[str, Any]:
        counts: dict[str, int] = {}
        invalidations: list[dict[str, Any]] = []
        for detail in self._iter_dashboard_details(
            report_detail_rows=report_detail_rows,
            interaction_detail_rows=interaction_detail_rows,
            scorecard_detail_rows=scorecard_detail_rows,
        ):
            lifecycle = detail.get("thesis_lifecycle")
            if isinstance(lifecycle, Mapping):
                stage = str(lifecycle.get("stage") or "").strip() or "unknown"
                counts[stage] = counts.get(stage, 0) + 1
            invalidation = detail.get("thesis_invalidation")
            if isinstance(invalidation, Mapping) and bool(invalidation.get("has_active_invalidation")):
                invalidations.append(
                    {
                        "summary": str(invalidation.get("summary") or "").strip() or "active invalidation",
                        "score": invalidation.get("score"),
                        "severity": invalidation.get("severity"),
                    }
                )
        count_rows = [{"stage": stage, "count": count} for stage, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))]
        invalidations.sort(key=lambda item: (float(item.get("score") or -9999.0), str(item.get("severity") or "")), reverse=True)
        return {"counts": count_rows[:6], "top_invalidations": invalidations[:5]}

    def _build_walk_forward_eval(self, *, scorecard_detail_rows: list[Any], window_size: int = 5) -> dict[str, Any]:
        closed_returns: list[float] = []
        for row in scorecard_detail_rows:
            if not isinstance(row, (list, tuple)) or len(row) < 3:
                continue
            if str(row[2] or "").strip().casefold() != "closed":
                continue
            detail = row[0]
            if not isinstance(detail, Mapping):
                continue
            try:
                value = float(detail.get("return_after_cost_pct")) if detail.get("return_after_cost_pct") is not None else float(row[1])
            except (TypeError, ValueError):
                continue
            closed_returns.append(value)
        if len(closed_returns) < max(3, window_size):
            return {"window_size": window_size, "window_count": 0, "avg_hit_rate_pct": None, "avg_return_after_cost_pct": None}
        windows: list[dict[str, Any]] = []
        for start in range(0, len(closed_returns) - window_size + 1):
            sample = closed_returns[start : start + window_size]
            hit_rate_pct = round((sum(1 for value in sample if value > 0) / len(sample)) * 100.0, 1)
            avg_return_after_cost_pct = round((sum(sample) / len(sample)) * 100.0, 2)
            windows.append(
                {
                    "index": len(windows) + 1,
                    "hit_rate_pct": hit_rate_pct,
                    "avg_return_after_cost_pct": avg_return_after_cost_pct,
                    "sample_count": len(sample),
                }
            )
        return {
            "window_size": window_size,
            "window_count": len(windows),
            "avg_hit_rate_pct": round(sum(float(item["hit_rate_pct"]) for item in windows) / len(windows), 1),
            "avg_return_after_cost_pct": round(sum(float(item["avg_return_after_cost_pct"]) for item in windows) / len(windows), 2),
            "windows": windows[-6:],
        }

    def _compute_avg_return_after_cost_pct(self, *, scorecard_detail_rows: list[Any]) -> float | None:
        return self._compute_avg_scorecard_detail_pct(
            scorecard_detail_rows=scorecard_detail_rows,
            field_name="return_after_cost_pct",
        )

    def _compute_avg_scorecard_detail_pct(self, *, scorecard_detail_rows: list[Any], field_name: str) -> float | None:
        values: list[float] = []
        for row in scorecard_detail_rows:
            if not isinstance(row, (list, tuple)) or len(row) < 3:
                continue
            if str(row[2] or "").strip().casefold() != "closed":
                continue
            detail = row[0]
            if not isinstance(detail, Mapping):
                continue
            try:
                value = float(detail.get(field_name))
            except (TypeError, ValueError):
                continue
            values.append(value * 100.0)
        if not values:
            return None
        return round(sum(values) / len(values), 2)

    @staticmethod
    def _iter_dashboard_details(
        *,
        report_detail_rows: list[Any],
        interaction_detail_rows: list[Any],
        scorecard_detail_rows: list[Any],
    ) -> list[Mapping[str, Any]]:
        details: list[Mapping[str, Any]] = []
        for row in report_detail_rows:
            detail = row[0] if isinstance(row, (list, tuple)) and row else row
            if isinstance(detail, Mapping):
                details.append(detail)
        for row in interaction_detail_rows:
            detail = row[0] if isinstance(row, (list, tuple)) and row else row
            if isinstance(detail, Mapping):
                details.append(detail)
        for row in scorecard_detail_rows:
            if isinstance(row, (list, tuple)) and row and isinstance(row[0], Mapping):
                details.append(row[0])
        return details

    def _build_source_ranking(
        self,
        *,
        report_detail_rows: list[Any],
        interaction_detail_rows: list[Any],
        scorecard_detail_rows: list[Any],
        thesis_stats: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        stats: dict[str, dict[str, Any]] = {}
        for row in report_detail_rows:
            detail = row[0] if isinstance(row, (list, tuple)) and row else row
            self._accumulate_source_stats(stats, detail=detail, bucket="reports")
        for row in interaction_detail_rows:
            detail = row[0] if isinstance(row, (list, tuple)) and row else row
            self._accumulate_source_stats(stats, detail=detail, bucket="interactions")
        for row in scorecard_detail_rows:
            if not isinstance(row, (list, tuple)) or len(row) < 3:
                continue
            self._accumulate_source_stats(
                stats,
                detail=row[0],
                bucket="scorecards",
                return_pct=row[1],
                status=row[2],
            )
        self._apply_thesis_weight_to_sources(
            stats=stats,
            scorecard_detail_rows=scorecard_detail_rows,
            thesis_stats=thesis_stats or {},
        )
        ranked: list[dict[str, Any]] = []
        for source, item in stats.items():
            closed_count = int(item.get("closed_count") or 0)
            return_count = int(item.get("return_count") or 0)
            avg_return_pct = (
                round((float(item.get("return_sum") or 0.0) / return_count) * 100.0, 2)
                if return_count
                else None
            )
            hit_rate_pct = round((int(item.get("win_count") or 0) / closed_count) * 100.0, 1) if closed_count else None
            thesis_weight_count = int(item.get("thesis_weight_count") or 0)
            thesis_alignment_pct = (
                round((float(item.get("thesis_weight_sum") or 0.0) / thesis_weight_count) * 100.0, 1)
                if thesis_weight_count
                else None
            )
            ttl_eval_count = int(item.get("ttl_eval_count") or 0)
            expired_count = int(item.get("expired_count") or 0)
            execution_observation_count = ttl_eval_count + int(item.get("fast_decay_count") or 0) + expired_count
            ttl_hit_rate_pct = (
                round((int(item.get("ttl_hit_count") or 0) / ttl_eval_count) * 100.0, 1)
                if ttl_eval_count
                else None
            )
            fast_decay_count = int(item.get("fast_decay_count") or 0)
            fast_decay_rate_pct = (
                round((fast_decay_count / closed_count) * 100.0, 1)
                if closed_count and execution_observation_count
                else None
            )
            hold_after_expiry_rate_pct = (
                round((int(item.get("hold_after_expiry_count") or 0) / expired_count) * 100.0, 1)
                if expired_count
                else None
            )
            ttl_fit_score = self._compute_ttl_fit_score(
                ttl_hit_rate_pct=ttl_hit_rate_pct,
                fast_decay_rate_pct=fast_decay_rate_pct,
                hold_after_expiry_rate_pct=hold_after_expiry_rate_pct,
                closed_count=closed_count,
            )
            source_health_count = int(item.get("source_health_count") or 0)
            source_health_score = (
                round((float(item.get("source_health_sum") or 0.0) / source_health_count), 1)
                if source_health_count
                else None
            )
            source_freshness_count = int(item.get("source_freshness_count") or 0)
            source_freshness_score = (
                round((float(item.get("source_freshness_sum") or 0.0) / source_freshness_count), 1)
                if source_freshness_count
                else None
            )
            best_ttl_bucket = self._select_best_ttl_bucket(item=item)
            mention_total = int(item.get("report_mentions") or 0) + int(item.get("interaction_mentions") or 0)
            weighted_score = self._compute_source_weighted_score(
                hit_rate_pct=hit_rate_pct,
                avg_return_pct=avg_return_pct,
                mention_total=mention_total,
                thesis_alignment_pct=thesis_alignment_pct,
                ttl_fit_score=ttl_fit_score,
                source_health_score=source_health_score,
            )
            ranked.append(
                {
                    "source": source,
                    "report_mentions": int(item.get("report_mentions") or 0),
                    "interaction_mentions": int(item.get("interaction_mentions") or 0),
                    "stock_pick_total": int(item.get("stock_pick_total") or 0),
                    "closed_count": closed_count,
                    "hit_rate_pct": hit_rate_pct,
                    "avg_return_pct": avg_return_pct,
                    "thesis_alignment_pct": thesis_alignment_pct,
                    "thesis_alignment": self._label_alignment(thesis_alignment_pct),
                    "ttl_hit_rate_pct": ttl_hit_rate_pct,
                    "fast_decay_rate_pct": fast_decay_rate_pct,
                    "hold_after_expiry_rate_pct": hold_after_expiry_rate_pct,
                    "ttl_fit_score": ttl_fit_score,
                    "best_ttl_bucket": best_ttl_bucket,
                    "source_health_score": source_health_score,
                    "source_freshness_score": source_freshness_score,
                    "weighted_score": weighted_score,
                }
            )
        ranked.sort(
            key=lambda item: (
                float(item.get("weighted_score") or -9999.0),
                int(item.get("closed_count") or 0),
                float(item.get("avg_return_pct") or -9999.0),
                int(item.get("report_mentions") or 0) + int(item.get("interaction_mentions") or 0),
                str(item.get("source") or ""),
            ),
            reverse=True,
        )
        return ranked[:12]

    def _build_thesis_stats(self, *, scorecard_detail_rows: list[Any]) -> dict[str, dict[str, Any]]:
        stats: dict[str, dict[str, Any]] = {}
        for row in scorecard_detail_rows:
            if not isinstance(row, (list, tuple)) or len(row) < 3:
                continue
            detail = row[0]
            if not isinstance(detail, Mapping):
                continue
            for label in self._extract_thesis_labels(detail):
                item = stats.setdefault(
                    label,
                    {
                        "stock_pick_total": 0,
                        "closed_count": 0,
                        "win_count": 0,
                        "return_sum": 0.0,
                        "return_count": 0,
                    },
                )
                item["stock_pick_total"] += 1
                if str(row[2] or "").strip().casefold() != "closed":
                    continue
                item["closed_count"] += 1
                try:
                    return_value = float(row[1]) if row[1] is not None else None
                except (TypeError, ValueError):
                    return_value = None
                if return_value is None:
                    continue
                item["return_sum"] += return_value
                item["return_count"] += 1
                if return_value > 0:
                    item["win_count"] += 1
        return stats

    def _serialize_thesis_ranking(self, *, thesis_stats: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
        ranked: list[dict[str, Any]] = []
        for thesis, item in thesis_stats.items():
            closed_count = int(item.get("closed_count") or 0)
            return_count = int(item.get("return_count") or 0)
            avg_return_pct = (
                round((float(item.get("return_sum") or 0.0) / return_count) * 100.0, 2)
                if return_count
                else None
            )
            hit_rate_pct = round((int(item.get("win_count") or 0) / closed_count) * 100.0, 1) if closed_count else None
            reliability_score = self._compute_thesis_reliability_score(
                closed_count=closed_count,
                hit_rate_pct=hit_rate_pct,
                avg_return_pct=avg_return_pct,
            )
            ranked.append(
                {
                    "thesis": thesis,
                    "stock_pick_total": int(item.get("stock_pick_total") or 0),
                    "closed_count": closed_count,
                    "hit_rate_pct": hit_rate_pct,
                    "avg_return_pct": avg_return_pct,
                    "reliability_score": reliability_score,
                }
            )
        ranked.sort(
            key=lambda item: (
                float(item.get("reliability_score") or -9999.0),
                int(item.get("closed_count") or 0),
                float(item.get("avg_return_pct") or -9999.0),
                int(item.get("stock_pick_total") or 0),
                str(item.get("thesis") or ""),
            ),
            reverse=True,
        )
        return ranked[:10]

    def _build_execution_panel(
        self,
        *,
        scorecard_detail_rows: list[Any],
        alert_kind_filter: str | None = None,
    ) -> dict[str, Any]:
        closed_postmortems = 0
        ttl_hit_count = 0
        ttl_eval_count = 0
        fast_decay_count = 0
        expired_count = 0
        hold_after_expiry_count = 0
        discard_after_expiry_count = 0
        ttl_kind_stats: dict[str, dict[str, dict[str, Any]]] = {}
        kind_stats: dict[str, dict[str, Any]] = {}
        normalized_filter = self._normalize_execution_alert_kind(alert_kind_filter)

        for row in scorecard_detail_rows:
            if not isinstance(row, (list, tuple)) or len(row) < 3:
                continue
            detail = row[0]
            if not isinstance(detail, Mapping):
                continue
            status = str(row[2] or "").strip().casefold()
            if status != "closed":
                continue
            alert_kind = str(detail.get("alert_kind") or "unknown").strip() or "unknown"
            if normalized_filter is not None and alert_kind != normalized_filter:
                continue
            closed_postmortems += 1
            kind_item = kind_stats.setdefault(
                alert_kind,
                {
                    "closed_postmortems": 0,
                    "ttl_eval_count": 0,
                    "ttl_hit_count": 0,
                    "fast_decay_count": 0,
                    "expired_count": 0,
                    "hold_after_expiry_count": 0,
                    "discard_after_expiry_count": 0,
                },
            )
            kind_item["closed_postmortems"] += 1
            raw_ttl_hit = detail.get("ttl_hit")
            if isinstance(raw_ttl_hit, bool):
                ttl_eval_count += 1
                if raw_ttl_hit:
                    ttl_hit_count += 1
                kind_item["ttl_eval_count"] += 1
                if raw_ttl_hit:
                    kind_item["ttl_hit_count"] += 1
            decay_label = str(detail.get("signal_decay_label") or "").strip()
            if decay_label == "fast_decay":
                fast_decay_count += 1
                kind_item["fast_decay_count"] += 1
            expired_before_evaluation = bool(detail.get("expired_before_evaluation"))
            postmortem_action = str(detail.get("postmortem_action") or "").strip()
            if expired_before_evaluation:
                expired_count += 1
                kind_item["expired_count"] += 1
                if postmortem_action == "hold_thesis":
                    hold_after_expiry_count += 1
                    kind_item["hold_after_expiry_count"] += 1
                elif postmortem_action == "discard_thesis":
                    discard_after_expiry_count += 1
                    kind_item["discard_after_expiry_count"] += 1

            ttl_bucket = self._ttl_bucket_label(detail.get("ttl_minutes"))
            bucket_stats = ttl_kind_stats.setdefault(alert_kind, {}).setdefault(
                ttl_bucket,
                {"sample_count": 0, "win_count": 0, "return_sum": 0.0, "return_count": 0, "hold_count": 0},
            )
            bucket_stats["sample_count"] += 1
            try:
                return_value = float(row[1]) if row[1] is not None else None
            except (TypeError, ValueError):
                return_value = None
            if return_value is not None:
                bucket_stats["return_sum"] += return_value
                bucket_stats["return_count"] += 1
                if return_value > 0:
                    bucket_stats["win_count"] += 1
            if postmortem_action == "hold_thesis":
                bucket_stats["hold_count"] += 1

        best_ttl_by_alert_kind: list[dict[str, Any]] = []
        for alert_kind, bucket_map in ttl_kind_stats.items():
            candidates: list[dict[str, Any]] = []
            for ttl_bucket, stats in bucket_map.items():
                sample_count = int(stats.get("sample_count") or 0)
                return_count = int(stats.get("return_count") or 0)
                avg_return_pct = (
                    round((float(stats.get("return_sum") or 0.0) / return_count) * 100.0, 2)
                    if return_count
                    else None
                )
                hit_rate_pct = (
                    round((int(stats.get("win_count") or 0) / sample_count) * 100.0, 1)
                    if sample_count
                    else None
                )
                hold_rate_pct = (
                    round((int(stats.get("hold_count") or 0) / sample_count) * 100.0, 1)
                    if sample_count
                    else None
                )
                score = self._compute_execution_ttl_score(
                    hit_rate_pct=hit_rate_pct,
                    avg_return_pct=avg_return_pct,
                    hold_rate_pct=hold_rate_pct,
                    sample_count=sample_count,
                )
                candidates.append(
                    {
                        "alert_kind": alert_kind,
                        "ttl_bucket": ttl_bucket,
                        "sample_count": sample_count,
                        "hit_rate_pct": hit_rate_pct,
                        "avg_return_pct": avg_return_pct,
                        "hold_rate_pct": hold_rate_pct,
                        "score": score,
                    }
                )
            if not candidates:
                continue
            candidates.sort(
                key=lambda item: (
                    float(item.get("score") or -9999.0),
                    int(item.get("sample_count") or 0),
                    float(item.get("avg_return_pct") or -9999.0),
                    str(item.get("ttl_bucket") or ""),
                ),
                reverse=True,
            )
            winner = dict(candidates[0])
            winner["best_ttl_bucket"] = winner.pop("ttl_bucket")
            best_ttl_by_alert_kind.append(winner)
        best_ttl_by_alert_kind.sort(
            key=lambda item: (
                float(item.get("score") or -9999.0),
                int(item.get("sample_count") or 0),
                str(item.get("alert_kind") or ""),
            ),
            reverse=True,
        )
        best_ttl_map = {
            str(item.get("alert_kind") or ""): item
            for item in best_ttl_by_alert_kind
            if isinstance(item, Mapping) and str(item.get("alert_kind") or "")
        }
        by_alert_kind: list[dict[str, Any]] = []
        for alert_kind, item in kind_stats.items():
            closed_count = int(item.get("closed_postmortems") or 0)
            ttl_eval_count_kind = int(item.get("ttl_eval_count") or 0)
            expired_count_kind = int(item.get("expired_count") or 0)
            best_ttl = best_ttl_map.get(alert_kind, {})
            by_alert_kind.append(
                {
                    "alert_kind": alert_kind,
                    "closed_postmortems": closed_count,
                    "ttl_hit_rate_pct": (
                        round((int(item.get("ttl_hit_count") or 0) / ttl_eval_count_kind) * 100.0, 1)
                        if ttl_eval_count_kind
                        else None
                    ),
                    "fast_decay_rate_pct": (
                        round((int(item.get("fast_decay_count") or 0) / closed_count) * 100.0, 1)
                        if closed_count
                        else None
                    ),
                    "hold_after_expiry_rate_pct": (
                        round((int(item.get("hold_after_expiry_count") or 0) / expired_count_kind) * 100.0, 1)
                        if expired_count_kind
                        else None
                    ),
                    "discard_after_expiry_rate_pct": (
                        round((int(item.get("discard_after_expiry_count") or 0) / expired_count_kind) * 100.0, 1)
                        if expired_count_kind
                        else None
                    ),
                    "best_ttl_bucket": best_ttl.get("best_ttl_bucket"),
                    "best_ttl_score": best_ttl.get("score"),
                    "best_ttl_sample_count": best_ttl.get("sample_count"),
                }
            )
        by_alert_kind.sort(
            key=lambda item: (
                float(item.get("best_ttl_score") or -9999.0),
                int(item.get("closed_postmortems") or 0),
                str(item.get("alert_kind") or ""),
            ),
            reverse=True,
        )

        return {
            "alert_kind_filter": normalized_filter,
            "closed_postmortems": closed_postmortems,
            "ttl_hit_rate_pct": round((ttl_hit_count / ttl_eval_count) * 100.0, 1) if ttl_eval_count else None,
            "fast_decay_rate_pct": round((fast_decay_count / closed_postmortems) * 100.0, 1) if closed_postmortems else None,
            "hold_after_expiry_rate_pct": (
                round((hold_after_expiry_count / expired_count) * 100.0, 1) if expired_count else None
            ),
            "discard_after_expiry_rate_pct": (
                round((discard_after_expiry_count / expired_count) * 100.0, 1) if expired_count else None
            ),
            "by_alert_kind": by_alert_kind,
            "best_ttl_by_alert_kind": best_ttl_by_alert_kind[:6],
            "source_ttl_heatmap": self._build_source_ttl_heatmap(
                scorecard_detail_rows=scorecard_detail_rows,
                alert_kind_filter=normalized_filter,
            ),
        }

    def _build_source_ttl_heatmap(
        self,
        *,
        scorecard_detail_rows: list[Any],
        alert_kind_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        heatmap: dict[tuple[str, str, str], dict[str, Any]] = {}
        normalized_filter = self._normalize_execution_alert_kind(alert_kind_filter)
        for row in scorecard_detail_rows:
            if not isinstance(row, (list, tuple)) or len(row) < 3:
                continue
            detail = row[0]
            if not isinstance(detail, Mapping):
                continue
            status = str(row[2] or "").strip().casefold()
            if status != "closed":
                continue
            coverage = detail.get("source_coverage")
            if not isinstance(coverage, Mapping):
                continue
            used_sources = coverage.get("used_sources")
            if not isinstance(used_sources, list):
                continue
            alert_kind = str(detail.get("alert_kind") or "unknown").strip() or "unknown"
            if normalized_filter is not None and alert_kind != normalized_filter:
                continue
            ttl_bucket = self._ttl_bucket_label(detail.get("ttl_minutes"))
            raw_ttl_hit = detail.get("ttl_hit")
            ttl_hit_value = raw_ttl_hit if isinstance(raw_ttl_hit, bool) else None
            postmortem_action = str(detail.get("postmortem_action") or "").strip()
            try:
                return_value = float(row[1]) if row[1] is not None else None
            except (TypeError, ValueError):
                return_value = None
            for source in dict.fromkeys(str(item).strip() for item in used_sources if str(item).strip()):
                key = (source, alert_kind, ttl_bucket)
                bucket = heatmap.setdefault(
                    key,
                    {
                        "source": source,
                        "alert_kind": alert_kind,
                        "ttl_bucket": ttl_bucket,
                        "sample_count": 0,
                        "ttl_eval_count": 0,
                        "ttl_hit_count": 0,
                        "hold_count": 0,
                        "return_sum": 0.0,
                        "return_count": 0,
                        "win_count": 0,
                    },
                )
                bucket["sample_count"] += 1
                if ttl_hit_value is not None:
                    bucket["ttl_eval_count"] += 1
                    if ttl_hit_value:
                        bucket["ttl_hit_count"] += 1
                if postmortem_action == "hold_thesis":
                    bucket["hold_count"] += 1
                if return_value is not None:
                    bucket["return_sum"] += return_value
                    bucket["return_count"] += 1
                    if return_value > 0:
                        bucket["win_count"] += 1

        rows: list[dict[str, Any]] = []
        for item in heatmap.values():
            sample_count = int(item.get("sample_count") or 0)
            ttl_eval_count = int(item.get("ttl_eval_count") or 0)
            return_count = int(item.get("return_count") or 0)
            hit_rate_pct = (
                round((int(item.get("win_count") or 0) / sample_count) * 100.0, 1)
                if sample_count
                else None
            )
            ttl_hit_rate_pct = (
                round((int(item.get("ttl_hit_count") or 0) / ttl_eval_count) * 100.0, 1)
                if ttl_eval_count
                else None
            )
            hold_rate_pct = (
                round((int(item.get("hold_count") or 0) / sample_count) * 100.0, 1)
                if sample_count
                else None
            )
            avg_return_pct = (
                round((float(item.get("return_sum") or 0.0) / return_count) * 100.0, 2)
                if return_count
                else None
            )
            score = self._compute_execution_ttl_score(
                hit_rate_pct=ttl_hit_rate_pct if ttl_hit_rate_pct is not None else hit_rate_pct,
                avg_return_pct=avg_return_pct,
                hold_rate_pct=hold_rate_pct,
                sample_count=sample_count,
            )
            rows.append(
                {
                    "source": item.get("source"),
                    "alert_kind": item.get("alert_kind"),
                    "ttl_bucket": item.get("ttl_bucket"),
                    "sample_count": sample_count,
                    "ttl_hit_rate_pct": ttl_hit_rate_pct,
                    "hit_rate_pct": hit_rate_pct,
                    "hold_rate_pct": hold_rate_pct,
                    "avg_return_pct": avg_return_pct,
                    "score": score,
                }
            )
        rows.sort(
            key=lambda item: (
                float(item.get("score") or -9999.0),
                int(item.get("sample_count") or 0),
                str(item.get("source") or ""),
                str(item.get("alert_kind") or ""),
                str(item.get("ttl_bucket") or ""),
            ),
            reverse=True,
        )
        return rows[:24]

    @staticmethod
    def _normalize_execution_alert_kind(value: str | None) -> str | None:
        normalized = str(value or "").strip().casefold().replace("-", "_")
        if normalized in {"stock_pick", "macro_playbook", "macro_surprise"}:
            return normalized
        return None

    @staticmethod
    def _accumulate_source_stats(
        stats: dict[str, dict[str, Any]],
        *,
        detail: Any,
        bucket: str,
        return_pct: Any = None,
        status: Any = None,
    ) -> None:
        if not isinstance(detail, Mapping):
            return
        coverage = detail.get("source_coverage")
        if not isinstance(coverage, Mapping):
            return
        used_sources = coverage.get("used_sources")
        if not isinstance(used_sources, list):
            return
        unique_sources = [
            str(item).strip()
            for item in used_sources
            if str(item).strip()
        ]
        for source in dict.fromkeys(unique_sources):
            item = stats.setdefault(
                source,
                {
                    "report_mentions": 0,
                    "interaction_mentions": 0,
                    "stock_pick_total": 0,
                    "closed_count": 0,
                    "win_count": 0,
                    "return_sum": 0.0,
                    "return_count": 0,
                    "source_health_sum": 0.0,
                    "source_health_count": 0,
                    "source_freshness_sum": 0.0,
                    "source_freshness_count": 0,
                },
            )
            source_health = detail.get("source_health")
            if isinstance(source_health, Mapping):
                try:
                    source_health_score = float(source_health.get("score"))
                except (TypeError, ValueError):
                    source_health_score = None
                try:
                    freshness_pct = float(source_health.get("freshness_pct"))
                except (TypeError, ValueError):
                    freshness_pct = None
                if source_health_score is not None:
                    item["source_health_sum"] = float(item.get("source_health_sum") or 0.0) + source_health_score
                    item["source_health_count"] = int(item.get("source_health_count") or 0) + 1
                if freshness_pct is not None:
                    item["source_freshness_sum"] = float(item.get("source_freshness_sum") or 0.0) + freshness_pct
                    item["source_freshness_count"] = int(item.get("source_freshness_count") or 0) + 1
            if bucket == "reports":
                item["report_mentions"] += 1
            elif bucket == "interactions":
                item["interaction_mentions"] += 1
            elif bucket == "scorecards":
                item["stock_pick_total"] += 1
                if str(status or "").strip().casefold() == "closed":
                    item["closed_count"] += 1
                    raw_ttl_hit = detail.get("ttl_hit")
                    if isinstance(raw_ttl_hit, bool):
                        item["ttl_eval_count"] = int(item.get("ttl_eval_count") or 0) + 1
                        if raw_ttl_hit:
                            item["ttl_hit_count"] = int(item.get("ttl_hit_count") or 0) + 1
                    decay_label = str(detail.get("signal_decay_label") or "").strip()
                    if decay_label == "fast_decay":
                        item["fast_decay_count"] = int(item.get("fast_decay_count") or 0) + 1
                    if bool(detail.get("expired_before_evaluation")):
                        item["expired_count"] = int(item.get("expired_count") or 0) + 1
                        postmortem_action = str(detail.get("postmortem_action") or "").strip()
                        if postmortem_action == "hold_thesis":
                            item["hold_after_expiry_count"] = int(item.get("hold_after_expiry_count") or 0) + 1
                        elif postmortem_action == "discard_thesis":
                            item["discard_after_expiry_count"] = int(item.get("discard_after_expiry_count") or 0) + 1
                    ttl_bucket = RuntimeHistoryStore._ttl_bucket_label(detail.get("ttl_minutes"))
                    if ttl_bucket != "unknown":
                        bucket_map = item.setdefault("ttl_bucket_stats", {})
                        bucket_stats = bucket_map.setdefault(
                            ttl_bucket,
                            {"sample_count": 0, "win_count": 0, "return_sum": 0.0, "return_count": 0, "hold_count": 0},
                        )
                        bucket_stats["sample_count"] += 1
                        if bool(detail.get("expired_before_evaluation")) and str(detail.get("postmortem_action") or "").strip() == "hold_thesis":
                            bucket_stats["hold_count"] += 1
                    if return_pct is not None:
                        try:
                            return_value = float(return_pct)
                        except (TypeError, ValueError):
                            return_value = None
                        if return_value is not None:
                            item["return_sum"] += return_value
                            item["return_count"] += 1
                            if return_value > 0:
                                item["win_count"] += 1
                            if ttl_bucket != "unknown":
                                bucket_map = item.setdefault("ttl_bucket_stats", {})
                                bucket_stats = bucket_map.setdefault(
                                    ttl_bucket,
                                    {"sample_count": 0, "win_count": 0, "return_sum": 0.0, "return_count": 0, "hold_count": 0},
                                )
                                bucket_stats["return_sum"] += return_value
                                bucket_stats["return_count"] += 1
                                if return_value > 0:
                                    bucket_stats["win_count"] += 1

    def _apply_thesis_weight_to_sources(
        self,
        *,
        stats: dict[str, dict[str, Any]],
        scorecard_detail_rows: list[Any],
        thesis_stats: Mapping[str, Mapping[str, Any]],
    ) -> None:
        if not thesis_stats:
            return
        thesis_reliability = {
            thesis: self._compute_thesis_reliability_score(
                closed_count=int(item.get("closed_count") or 0),
                hit_rate_pct=round((int(item.get("win_count") or 0) / int(item.get("closed_count") or 1)) * 100.0, 1)
                if int(item.get("closed_count") or 0)
                else None,
                avg_return_pct=(
                    round((float(item.get("return_sum") or 0.0) / int(item.get("return_count") or 1)) * 100.0, 2)
                    if int(item.get("return_count") or 0)
                    else None
                ),
            )
            for thesis, item in thesis_stats.items()
        }
        for row in scorecard_detail_rows:
            if not isinstance(row, (list, tuple)) or len(row) < 3:
                continue
            detail = row[0]
            if not isinstance(detail, Mapping):
                continue
            coverage = detail.get("source_coverage")
            if not isinstance(coverage, Mapping):
                continue
            used_sources = coverage.get("used_sources")
            if not isinstance(used_sources, list):
                continue
            thesis_labels = self._extract_thesis_labels(detail)
            if not thesis_labels:
                continue
            matched_scores = [float(thesis_reliability[label]) for label in thesis_labels if label in thesis_reliability]
            if not matched_scores:
                continue
            alignment_score = sum(matched_scores) / len(matched_scores)
            for source in dict.fromkeys(str(item).strip() for item in used_sources if str(item).strip()):
                item = stats.setdefault(
                    source,
                    {
                        "report_mentions": 0,
                        "interaction_mentions": 0,
                        "stock_pick_total": 0,
                        "closed_count": 0,
                        "win_count": 0,
                        "return_sum": 0.0,
                        "return_count": 0,
                        "thesis_weight_sum": 0.0,
                        "thesis_weight_count": 0,
                    },
                )
                item["thesis_weight_sum"] = float(item.get("thesis_weight_sum") or 0.0) + alignment_score
                item["thesis_weight_count"] = int(item.get("thesis_weight_count") or 0) + 1

    @staticmethod
    def _compute_thesis_reliability_score(
        *,
        closed_count: int,
        hit_rate_pct: float | None,
        avg_return_pct: float | None,
    ) -> float:
        if closed_count <= 0:
            return 0.5
        hit_component = max(-0.25, min(0.35, ((float(hit_rate_pct or 50.0) - 50.0) / 100.0)))
        return_component = max(-0.15, min(0.2, float(avg_return_pct or 0.0) / 20.0))
        sample_component = min(0.1, closed_count * 0.02)
        score = 0.5 + hit_component + return_component + sample_component
        return round(max(0.05, min(0.98, score)), 3)

    @staticmethod
    def _compute_source_weighted_score(
        *,
        hit_rate_pct: float | None,
        avg_return_pct: float | None,
        mention_total: int,
        thesis_alignment_pct: float | None,
        ttl_fit_score: float | None,
        source_health_score: float | None,
    ) -> float:
        base_score = (
            (float(hit_rate_pct or 0.0) * 0.45)
            + (max(-10.0, min(12.0, float(avg_return_pct or 0.0))) * 2.0)
            + min(15.0, max(0, mention_total) * 1.2)
        )
        if thesis_alignment_pct is not None:
            base_score = (base_score * 0.65) + (float(thesis_alignment_pct) * 0.35)
        if ttl_fit_score is not None:
            base_score = (base_score * 0.75) + (float(ttl_fit_score) * 0.25)
        if source_health_score is not None:
            base_score = (base_score * 0.82) + (float(source_health_score) * 0.18)
        return round(base_score, 1)

    @staticmethod
    def _compute_ttl_fit_score(
        *,
        ttl_hit_rate_pct: float | None,
        fast_decay_rate_pct: float | None,
        hold_after_expiry_rate_pct: float | None,
        closed_count: int,
    ) -> float | None:
        if ttl_hit_rate_pct is None and fast_decay_rate_pct is None and hold_after_expiry_rate_pct is None:
            return None
        score = (
            (float(ttl_hit_rate_pct or 0.0) * 0.5)
            + (float(hold_after_expiry_rate_pct or 0.0) * 0.35)
            - (float(fast_decay_rate_pct or 0.0) * 0.45)
            + min(10.0, max(0, closed_count) * 1.0)
        )
        return round(max(0.0, min(100.0, score)), 1)

    def _select_best_ttl_bucket(self, *, item: Mapping[str, Any]) -> str | None:
        bucket_map = item.get("ttl_bucket_stats")
        if not isinstance(bucket_map, Mapping):
            return None
        candidates: list[tuple[float, int, str]] = []
        for ttl_bucket, raw_stats in bucket_map.items():
            if not isinstance(ttl_bucket, str) or not isinstance(raw_stats, Mapping):
                continue
            sample_count = int(raw_stats.get("sample_count") or 0)
            if sample_count <= 0:
                continue
            return_count = int(raw_stats.get("return_count") or 0)
            avg_return_pct = (
                round((float(raw_stats.get("return_sum") or 0.0) / return_count) * 100.0, 2)
                if return_count
                else None
            )
            hit_rate_pct = round((int(raw_stats.get("win_count") or 0) / sample_count) * 100.0, 1)
            hold_rate_pct = round((int(raw_stats.get("hold_count") or 0) / sample_count) * 100.0, 1)
            score = self._compute_execution_ttl_score(
                hit_rate_pct=hit_rate_pct,
                avg_return_pct=avg_return_pct,
                hold_rate_pct=hold_rate_pct,
                sample_count=sample_count,
            )
            candidates.append((score, sample_count, ttl_bucket))
        if not candidates:
            return None
        candidates.sort(reverse=True)
        return candidates[0][2]

    @staticmethod
    def _label_alignment(thesis_alignment_pct: float | None) -> str | None:
        if thesis_alignment_pct is None:
            return None
        if thesis_alignment_pct >= 72:
            return "high"
        if thesis_alignment_pct >= 58:
            return "medium"
        return "low"

    @staticmethod
    def _extract_thesis_labels(detail: Mapping[str, Any]) -> list[str]:
        labels: list[str] = []
        thesis_summary = str(detail.get("thesis_summary") or "").strip()
        if thesis_summary:
            labels.append(RuntimeHistoryStore._normalize_thesis_label(thesis_summary))
        thesis_memory = detail.get("thesis_memory")
        if isinstance(thesis_memory, list):
            for item in thesis_memory[:2]:
                if not isinstance(item, Mapping):
                    continue
                thesis_text = str(item.get("thesis_text") or "").strip()
                if thesis_text:
                    labels.append(RuntimeHistoryStore._normalize_thesis_label(thesis_text))
        macro_headline = str(detail.get("macro_headline") or "").strip()
        if not labels and macro_headline:
            labels.append(RuntimeHistoryStore._normalize_thesis_label(macro_headline))
        unique = [label for label in dict.fromkeys(labels) if label]
        return unique[:2]

    @staticmethod
    def _normalize_thesis_label(text: str, *, limit: int = 88) -> str:
        normalized = " ".join(str(text or "").split())
        if len(normalized) <= limit:
            return normalized
        return normalized[: limit - 3].rstrip() + "..."

    @staticmethod
    def _ttl_bucket_label(raw_ttl_minutes: Any) -> str:
        try:
            ttl_minutes = float(raw_ttl_minutes)
        except (TypeError, ValueError):
            return "unknown"
        if ttl_minutes <= 120:
            return "short"
        if ttl_minutes <= 240:
            return "medium"
        return "long"

    @staticmethod
    def _compute_execution_ttl_score(
        *,
        hit_rate_pct: float | None,
        avg_return_pct: float | None,
        hold_rate_pct: float | None,
        sample_count: int,
    ) -> float:
        base = (
            (float(hit_rate_pct or 0.0) * 0.45)
            + (max(-10.0, min(12.0, float(avg_return_pct or 0.0))) * 2.2)
            + (float(hold_rate_pct or 0.0) * 0.2)
            + min(12.0, max(0, sample_count) * 1.2)
        )
        return round(base, 1)

    def cleanup_retention(self) -> None:
        if self._db is None:
            return
        self._cleanup_table("bot_provider_diagnostics_history", "event_at")
        self._cleanup_table("bot_job_diagnostics_history", "event_at")
        self._cleanup_table("bot_sent_reports_history", "sent_at")
        self._cleanup_table("bot_user_interaction_history", "occurred_at")
        self._cleanup_table("bot_alert_audit", "sent_at")
        self._cleanup_table("bot_stock_pick_scorecard", "created_at")
        self._cleanup_table("bot_thesis_memory", "created_at")
        self._cleanup_table("bot_eval_artifact", "created_at")

    def _cleanup_table(self, table_name: str, time_column: str) -> None:
        if self._db is None:
            return
        self._db.execute(
            f"DELETE FROM {table_name} WHERE {time_column} < NOW() - (%s * INTERVAL '1 day')",
            (self.retention_days,),
        )

    @staticmethod
    def _embed_text(text: str, *, dimensions: int = 16) -> list[float]:
        vector = [0.0] * max(4, dimensions)
        tokens = [token for token in re.findall(r"[A-Za-z0-9_]+", (text or "").casefold()) if token]
        if not tokens:
            return vector
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            for index in range(len(vector)):
                vector[index] += (digest[index % len(digest)] / 255.0) - 0.5
        norm = math.sqrt(sum(value * value for value in vector))
        if norm <= 0:
            return vector
        return [round(value / norm, 6) for value in vector]

    @staticmethod
    def _vector_literal(values: list[float]) -> str:
        return "[" + ",".join(f"{float(value):.6f}" for value in values) + "]"
