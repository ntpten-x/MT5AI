from __future__ import annotations

import http.client
import json
import threading
from http.server import HTTPServer

from invest_advisor_bot.bot.health_check import HealthCheckHandler, set_health_details_provider


def _start_test_server() -> tuple[HTTPServer, threading.Thread]:
    server = HTTPServer(("127.0.0.1", 0), HealthCheckHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def test_health_endpoint_supports_get() -> None:
    server, thread = _start_test_server()
    try:
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
        connection.request("GET", "/health")
        response = connection.getresponse()
        body = response.read()
        connection.close()

        assert response.status == 200
        assert response.getheader("Content-Type") == "application/json; charset=utf-8"
        assert json.loads(body.decode("utf-8")) == {"status": "ok", "service": "invest-advisor-bot"}
    finally:
        set_health_details_provider(None)
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_health_endpoint_supports_head_without_body() -> None:
    server, thread = _start_test_server()
    try:
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
        connection.request("HEAD", "/health")
        response = connection.getresponse()
        body = response.read()
        connection.close()

        assert response.status == 200
        assert response.getheader("Content-Type") == "application/json; charset=utf-8"
        assert response.getheader("Content-Length") == str(len(json.dumps({"status": "ok", "service": "invest-advisor-bot"}).encode("utf-8")))
        assert body == b""
    finally:
        set_health_details_provider(None)
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_health_endpoint_includes_mlflow_details_when_provider_is_set() -> None:
    set_health_details_provider(
        lambda: {
            "mlflow": {
                "enabled": False,
                "tracking_configured": True,
                "warning": "mlflow_dependency_missing: install with pip install -e .[observability]",
                "last_run_id": None,
            }
        }
    )
    server, thread = _start_test_server()
    try:
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
        connection.request("GET", "/health")
        response = connection.getresponse()
        body = response.read()
        connection.close()

        payload = json.loads(body.decode("utf-8"))
        assert response.status == 200
        assert payload["status"] == "ok"
        assert payload["mlflow"]["enabled"] is False
        assert payload["mlflow"]["tracking_configured"] is True
        assert "mlflow_dependency_missing" in payload["mlflow"]["warning"]
    finally:
        set_health_details_provider(None)
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_diagnostics_endpoint_returns_full_runtime_payload() -> None:
    set_health_details_provider(
        lambda: {
            "mlflow": {"enabled": True},
            "diagnostics": {
                "status": "ok",
                "service": "invest-advisor-bot",
                "runtime": {
                    "started_at": "2026-03-24T11:00:00+00:00",
                    "uptime_seconds": 42.5,
                    "services": {
                        "llm_client": {"available": True, "provider_order": ["gemini", "groq"]},
                    },
                },
            },
        }
    )
    server, thread = _start_test_server()
    try:
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
        connection.request("GET", "/diagnostics")
        response = connection.getresponse()
        body = response.read()
        connection.close()

        payload = json.loads(body.decode("utf-8"))
        assert response.status == 200
        assert payload["status"] == "ok"
        assert payload["runtime"]["uptime_seconds"] == 42.5
        assert payload["runtime"]["services"]["llm_client"]["provider_order"] == ["gemini", "groq"]
        assert "mlflow" not in payload
    finally:
        set_health_details_provider(None)
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_metrics_endpoint_returns_prometheus_text() -> None:
    set_health_details_provider(
        lambda: {
            "diagnostics": {
                "status": "ok",
                "service": "invest-advisor-bot",
                "runtime": {
                    "uptime_seconds": 42.5,
                    "alerts_today": {"total": 3, "by_category": {"risk": 2}},
                    "db_state": {"healthy": False},
                    "provider_circuit": {"groq": {"is_open": True, "failure_count": 3}},
                    "provider_latency": [
                        {
                            "service": "llm_client",
                            "provider": "groq",
                            "operation": "generate_text",
                            "last_latency_ms": 150.0,
                            "avg_latency_ms": 140.0,
                            "p95_latency_ms": 190.0,
                            "p99_latency_ms": 198.0,
                            "window_size": 20,
                            "sample_count": 3,
                            "success_count": 2,
                            "failure_count": 1,
                        }
                    ],
                    "mlflow": {"enabled": True, "tracking_configured": True, "warning": "dependency missing"},
                    "response_stats": {"recommendation_service": {"total": 10, "fallback": 2, "fallback_rate": 20.0}},
                    "jobs": {"risk_alert_monitor": {"success_count": 5, "failure_count": 1, "duration_ms": 1200}},
                    "services": {"llm_client": {"available": True}},
                },
            }
        }
    )
    server, thread = _start_test_server()
    try:
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
        connection.request("GET", "/metrics")
        response = connection.getresponse()
        body = response.read().decode("utf-8")
        connection.close()

        assert response.status == 200
        assert response.getheader("Content-Type") == "text/plain; version=0.0.4; charset=utf-8"
        assert "invest_advisor_bot_uptime_seconds 42.5" in body
        assert 'invest_advisor_bot_provider_circuit_open{provider="groq"} 1' in body
        assert 'invest_advisor_bot_provider_latency_last_ms{operation="generate_text",provider="groq",service="llm_client"} 150.0' in body
        assert 'invest_advisor_bot_provider_latency_p95_ms{operation="generate_text",provider="groq",service="llm_client"} 190.0' in body
        assert "invest_advisor_bot_mlflow_warning 1" in body
        assert 'invest_advisor_bot_service_available{service="llm_client"} 1' in body
    finally:
        set_health_details_provider(None)
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
