from __future__ import annotations

from pathlib import Path

from invest_advisor_bot.bot.runtime_history_store import RuntimeHistoryStore


def test_runtime_history_store_weights_source_ranking_with_thesis_reliability(tmp_path: Path) -> None:
    store = RuntimeHistoryStore(path=tmp_path / "runtime.json")
    scorecard_detail_rows = [
        (
            {
                "source_coverage": {"used_sources": ["fred", "bls"]},
                "thesis_summary": "sticky inflation lowers the setup for duration-sensitive growth",
            },
            0.08,
            "closed",
        ),
        (
            {
                "source_coverage": {"used_sources": ["fred", "bea"]},
                "thesis_summary": "sticky inflation lowers the setup for duration-sensitive growth",
            },
            0.04,
            "closed",
        ),
        (
            {
                "source_coverage": {"used_sources": ["bea"]},
                "thesis_summary": "growth slowdown hurts cyclicals first",
            },
            -0.03,
            "closed",
        ),
    ]

    thesis_stats = store._build_thesis_stats(scorecard_detail_rows=scorecard_detail_rows)  # type: ignore[attr-defined]
    ranked = store._build_source_ranking(  # type: ignore[attr-defined]
        report_detail_rows=[],
        interaction_detail_rows=[],
        scorecard_detail_rows=scorecard_detail_rows,
        thesis_stats=thesis_stats,
    )

    fred = next(item for item in ranked if item["source"] == "fred")
    bea = next(item for item in ranked if item["source"] == "bea")
    assert fred["weighted_score"] > bea["weighted_score"]
    assert fred["thesis_alignment"] == "high"
    assert fred["ttl_fit_score"] is None


def test_runtime_history_store_builds_execution_panel_with_best_ttl(tmp_path: Path) -> None:
    store = RuntimeHistoryStore(path=tmp_path / "runtime.json")
    scorecard_detail_rows = [
        (
            {
                "alert_kind": "stock_pick",
                "ttl_minutes": 90,
                "ttl_hit": True,
                "signal_decay_label": "durable",
                "postmortem_action": "hold_thesis",
                "expired_before_evaluation": True,
            },
            0.05,
            "closed",
        ),
        (
            {
                "alert_kind": "stock_pick",
                "ttl_minutes": 95,
                "ttl_hit": True,
                "signal_decay_label": "durable",
                "postmortem_action": "hold_thesis",
                "expired_before_evaluation": True,
            },
            0.04,
            "closed",
        ),
        (
            {
                "alert_kind": "stock_pick",
                "ttl_minutes": 300,
                "ttl_hit": False,
                "signal_decay_label": "fast_decay",
                "postmortem_action": "discard_thesis",
                "expired_before_evaluation": True,
            },
            -0.03,
            "closed",
        ),
        (
            {
                "alert_kind": "macro_playbook",
                "ttl_minutes": 180,
                "ttl_hit": False,
                "signal_decay_label": "late_follow_through",
                "postmortem_action": "hold_thesis",
                "expired_before_evaluation": True,
            },
            0.03,
            "closed",
        ),
    ]

    panel = store._build_execution_panel(scorecard_detail_rows=scorecard_detail_rows)  # type: ignore[attr-defined]

    assert panel["closed_postmortems"] == 4
    assert panel["ttl_hit_rate_pct"] == 50.0
    assert panel["fast_decay_rate_pct"] == 25.0
    assert panel["hold_after_expiry_rate_pct"] == 75.0
    assert panel["source_ttl_heatmap"] == []
    assert panel["by_alert_kind"]
    stock_pick_best = next(item for item in panel["best_ttl_by_alert_kind"] if item["alert_kind"] == "stock_pick")
    assert stock_pick_best["best_ttl_bucket"] == "short"


def test_runtime_history_store_source_ranking_includes_ttl_fit_metrics(tmp_path: Path) -> None:
    store = RuntimeHistoryStore(path=tmp_path / "runtime.json")
    scorecard_detail_rows = [
        (
            {
                "source_coverage": {"used_sources": ["fred", "treasury"]},
                "source_health": {"score": 82.0, "freshness_pct": 88.0},
                "ttl_minutes": 90,
                "ttl_hit": True,
                "signal_decay_label": "durable",
                "postmortem_action": "hold_thesis",
                "expired_before_evaluation": True,
            },
            0.05,
            "closed",
        ),
        (
            {
                "source_coverage": {"used_sources": ["fred"]},
                "source_health": {"score": 78.0, "freshness_pct": 82.0},
                "ttl_minutes": 100,
                "ttl_hit": True,
                "signal_decay_label": "durable",
                "postmortem_action": "hold_thesis",
                "expired_before_evaluation": True,
            },
            0.03,
            "closed",
        ),
        (
            {
                "source_coverage": {"used_sources": ["bea"]},
                "source_health": {"score": 44.0, "freshness_pct": 40.0},
                "ttl_minutes": 300,
                "ttl_hit": False,
                "signal_decay_label": "fast_decay",
                "postmortem_action": "discard_thesis",
                "expired_before_evaluation": True,
            },
            -0.04,
            "closed",
        ),
    ]

    ranked = store._build_source_ranking(  # type: ignore[attr-defined]
        report_detail_rows=[],
        interaction_detail_rows=[],
        scorecard_detail_rows=scorecard_detail_rows,
        thesis_stats={},
    )

    fred = next(item for item in ranked if item["source"] == "fred")
    bea = next(item for item in ranked if item["source"] == "bea")
    assert fred["ttl_fit_score"] > bea["ttl_fit_score"]
    assert fred["best_ttl_bucket"] == "short"
    assert fred["ttl_hit_rate_pct"] == 100.0
    assert fred["source_health_score"] > bea["source_health_score"]
    assert fred["source_freshness_score"] > bea["source_freshness_score"]


def test_runtime_history_store_builds_source_ttl_heatmap(tmp_path: Path) -> None:
    store = RuntimeHistoryStore(path=tmp_path / "runtime.json")
    scorecard_detail_rows = [
        (
            {
                "source_coverage": {"used_sources": ["fred", "bea"]},
                "alert_kind": "stock_pick",
                "ttl_minutes": 90,
                "ttl_hit": True,
                "postmortem_action": "hold_thesis",
            },
            0.05,
            "closed",
        ),
        (
            {
                "source_coverage": {"used_sources": ["fred"]},
                "alert_kind": "stock_pick",
                "ttl_minutes": 210,
                "ttl_hit": False,
                "postmortem_action": "discard_thesis",
            },
            -0.01,
            "closed",
        ),
    ]

    heatmap = store._build_source_ttl_heatmap(scorecard_detail_rows=scorecard_detail_rows)  # type: ignore[attr-defined]

    fred_short = next(item for item in heatmap if item["source"] == "fred" and item["ttl_bucket"] == "short")
    assert fred_short["alert_kind"] == "stock_pick"
    assert fred_short["ttl_hit_rate_pct"] == 100.0


def test_runtime_history_store_builds_decision_quality_snapshot(tmp_path: Path) -> None:
    store = RuntimeHistoryStore(path=tmp_path / "runtime.json")
    snapshot = store._build_decision_quality_snapshot(  # type: ignore[attr-defined]
        report_detail_rows=[
            (
                {
                    "source_health": {"score": 82.0, "freshness_pct": 88.0},
                    "no_trade_decision": {"should_abstain": False, "reasons": []},
                },
            )
        ],
        interaction_detail_rows=[
            (
                {
                    "source_health": {"score": 46.0, "freshness_pct": 41.0},
                    "no_trade_decision": {
                        "should_abstain": True,
                        "reasons": ["supporting data coverage is still too thin"],
                    },
                },
            )
        ],
        scorecard_detail_rows=[
            (
                {
                    "source_health": {"score": 64.0, "freshness_pct": 72.0},
                    "no_trade_decision": {
                        "should_abstain": True,
                        "reasons": ["supporting data coverage is still too thin"],
                    },
                },
                0.01,
                "closed",
            )
        ],
    )

    assert snapshot["source_health"]["sample_count"] == 3
    assert snapshot["source_health"]["strong_count"] == 1
    assert snapshot["source_health"]["fragile_count"] == 1
    assert snapshot["source_health"]["degraded_sla_count"] == 0
    assert snapshot["no_trade"]["decision_count"] == 3
    assert snapshot["no_trade"]["abstain_count"] == 2
    assert snapshot["no_trade"]["top_reasons"][0]["reason"] == "supporting data coverage is still too thin"


def test_runtime_history_store_computes_average_return_after_cost(tmp_path: Path) -> None:
    store = RuntimeHistoryStore(path=tmp_path / "runtime.json")
    avg_return_after_cost_pct = store._compute_avg_return_after_cost_pct(  # type: ignore[attr-defined]
        scorecard_detail_rows=[
            ({"return_after_cost_pct": 0.041}, 0.05, "closed"),
            ({"return_after_cost_pct": -0.012}, -0.01, "closed"),
        ]
    )

    assert avg_return_after_cost_pct == 1.45


def test_runtime_history_store_computes_average_alpha_after_cost(tmp_path: Path) -> None:
    store = RuntimeHistoryStore(path=tmp_path / "runtime.json")
    avg_alpha_after_cost_pct = store._compute_avg_scorecard_detail_pct(  # type: ignore[attr-defined]
        scorecard_detail_rows=[
            ({"alpha_after_cost_pct": 0.021}, 0.05, "closed"),
            ({"alpha_after_cost_pct": -0.006}, -0.01, "closed"),
        ],
        field_name="alpha_after_cost_pct",
    )

    assert avg_alpha_after_cost_pct == 0.75


def test_runtime_history_store_builds_walk_forward_eval(tmp_path: Path) -> None:
    store = RuntimeHistoryStore(path=tmp_path / "runtime.json")
    snapshot = store._build_walk_forward_eval(  # type: ignore[attr-defined]
        scorecard_detail_rows=[
            ({"return_after_cost_pct": 0.03}, 0.03, "closed"),
            ({"return_after_cost_pct": 0.01}, 0.01, "closed"),
            ({"return_after_cost_pct": -0.02}, -0.02, "closed"),
            ({"return_after_cost_pct": 0.04}, 0.04, "closed"),
            ({"return_after_cost_pct": 0.02}, 0.02, "closed"),
        ]
    )

    assert snapshot["window_count"] == 1
    assert snapshot["avg_hit_rate_pct"] == 80.0


def test_runtime_history_store_filters_execution_panel_by_alert_kind(tmp_path: Path) -> None:
    store = RuntimeHistoryStore(path=tmp_path / "runtime.json")
    scorecard_detail_rows = [
        (
            {
                "source_coverage": {"used_sources": ["fred"]},
                "alert_kind": "stock_pick",
                "ttl_minutes": 90,
                "ttl_hit": True,
                "postmortem_action": "hold_thesis",
                "expired_before_evaluation": True,
            },
            0.05,
            "closed",
        ),
        (
            {
                "source_coverage": {"used_sources": ["macro_surprise_engine"]},
                "alert_kind": "macro_surprise",
                "ttl_minutes": 75,
                "ttl_hit": False,
                "postmortem_action": "discard_thesis",
                "expired_before_evaluation": True,
            },
            -0.01,
            "closed",
        ),
    ]

    panel = store._build_execution_panel(  # type: ignore[attr-defined]
        scorecard_detail_rows=scorecard_detail_rows,
        alert_kind_filter="stock_pick",
    )

    assert panel["alert_kind_filter"] == "stock_pick"
    assert panel["closed_postmortems"] == 1
    assert all(item["alert_kind"] == "stock_pick" for item in panel["source_ttl_heatmap"])
