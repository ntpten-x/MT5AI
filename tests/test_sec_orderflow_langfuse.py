from __future__ import annotations

from pathlib import Path

from invest_advisor_bot.dbt_semantic_layer import DbtSemanticLayer
from invest_advisor_bot.human_review_store import HumanReviewStore
from invest_advisor_bot.langfuse_observer import LangfuseObserver
from invest_advisor_bot.providers.order_flow_client import OrderFlowClient
from invest_advisor_bot.providers.ownership_client import OwnershipIntelligenceClient
from invest_advisor_bot.providers.policy_feed_client import PolicyFeedClient


def test_order_flow_client_parses_generic_payload() -> None:
    client = OrderFlowClient(enabled=True, api_key="key", base_url="https://flow.example.test")

    snapshot = client._parse_order_flow_payload(
        symbol="AAPL",
        payload={
            "data": {
                "bullish_premium": 1250000,
                "bearish_premium": 410000,
                "call_put_ratio": 1.9,
                "unusual_count": 8,
                "sweep_count": 3,
                "opening_flow_ratio": 0.74,
            }
        },
    )

    assert snapshot is not None
    assert snapshot.symbol == "AAPL"
    assert snapshot.sentiment == "bullish"
    assert snapshot.unusual_count == 8


def test_policy_feed_client_parses_rss_feed() -> None:
    client = PolicyFeedClient(enabled=True)

    events = client._parse_feed(
        text="""
        <rss>
          <channel>
            <item>
              <title>Federal Reserve Chair remarks on inflation outlook</title>
              <link>https://example.test/fed-speech</link>
              <pubDate>Tue, 25 Mar 2026 10:30:00 +0000</pubDate>
              <description>Price stability remains the focus.</description>
            </item>
          </channel>
        </rss>
        """,
        central_bank="fed",
        category="speech",
    )

    assert len(events) == 1
    assert events[0].central_bank == "fed"
    assert events[0].tone_signal == "hawkish"


def test_ownership_client_extracts_13d_and_13f(monkeypatch) -> None:
    client = OwnershipIntelligenceClient(
        sec_user_agent="InvestAdvisorBot/0.2 support@example.com",
        manager_ciks=("0001067983",),
    )

    monkeypatch.setattr(
        client,
        "_load_sec_ticker_mapping",
        lambda: {"AAPL": {"cik": "0000320193", "title": "Apple Inc."}},
    )

    def fake_fetch_sec_json(url: str):
        if "CIK0000320193.json" in url:
            return {
                "filings": {
                    "recent": {
                        "form": ["SC 13D"],
                        "filingDate": ["2026-03-20"],
                        "accessionNumber": ["0000320193-26-000010"],
                        "primaryDocument": ["aapl13d.htm"],
                    }
                }
            }
        if "CIK0001067983.json" in url:
            return {
                "name": "Berkshire Hathaway",
                "filings": {
                    "recent": {
                        "form": ["13F-HR"],
                        "filingDate": ["2026-02-14"],
                        "accessionNumber": ["0001067983-26-000001"],
                        "primaryDocument": ["primary_doc.xml"],
                    }
                },
            }
        return None

    def fake_fetch_text(url: str):
        if "aapl13d.htm" in url:
            return """
            NAME OF REPORTING PERSON
            Berkshire Hathaway Inc.
            AGGREGATE AMOUNT BENEFICIALLY OWNED BY EACH REPORTING PERSON 915,560,382
            PERCENT OF CLASS REPRESENTED BY AMOUNT IN ROW (11) 5.70%
            """
        if "primary_doc.xml" in url:
            return """
            <html><body><a href="infotable.xml">information table</a></body></html>
            """
        if "infotable.xml" in url:
            return """
            <informationTable>
              <infoTable>
                <nameOfIssuer>APPLE INC</nameOfIssuer>
                <value>174321000</value>
                <shrsOrPrnAmt><sshPrnamt>915560382</sshPrnamt></shrsOrPrnAmt>
              </infoTable>
            </informationTable>
            """
        return None

    monkeypatch.setattr(client, "_fetch_sec_json", fake_fetch_sec_json)
    monkeypatch.setattr(client, "_fetch_text", fake_fetch_text)

    ownership = client._get_company_ownership_sync("AAPL", "Apple Inc.")

    assert ownership is not None
    assert ownership.ownership_signal in {"meaningful_beneficial_owner", "activist_or_anchor_holder"}
    assert ownership.beneficial_owners[0].stake_pct == 5.7
    assert ownership.institutional_holders[0].manager_name == "Berkshire Hathaway"


def test_langfuse_observer_and_human_review_store(tmp_path: Path) -> None:
    observer = LangfuseObserver(root_dir=tmp_path / "langfuse", enabled=True)
    observer.log_recommendation(
        artifact_key="artifact-1",
        question="What should I buy?",
        response_text="Consider quality growth.",
        model="gpt-test",
        fallback_used=False,
        payload={"market_confidence": {"score": 0.51}},
        data_quality={"blocking": False},
    )

    status = observer.status()
    assert status["event_count"] == 1
    assert (tmp_path / "langfuse" / "langfuse_recommendations.jsonl").exists()

    store = HumanReviewStore(root_dir=tmp_path / "reviews", enabled=True, sample_every_n=2)
    review_id = store.enqueue(
        artifact_key="artifact-1",
        question="What should I buy?",
        recommendation_text="Consider quality growth.",
        model="gpt-test",
        fallback_used=False,
        confidence_score=0.41,
        metadata={"service_name": "recommendation_service"},
    )

    assert review_id is not None
    assert len(store.list_pending()) == 1
    completed = store.complete_review(review_id=review_id, decision="accepted", score=0.8, note="looks good")
    assert completed is not None
    assert completed["decision"] == "accepted"
    assert len(store.list_pending()) == 0


def test_dbt_semantic_layer_sync_writes_project(tmp_path: Path) -> None:
    layer = DbtSemanticLayer(
        root_dir=tmp_path / "dbt_semantic",
        enabled=True,
        project_name="advisor_semantic",
        target_schema="analytics_prod",
    )

    layer.sync()
    status = layer.status()

    assert status["enabled"] is True
    assert (tmp_path / "dbt_semantic" / "dbt_project.yml").exists()
    assert (tmp_path / "dbt_semantic" / "models" / "marts" / "semantic_metrics.yml").exists()
