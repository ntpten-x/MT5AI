from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Callable, Mapping

from loguru import logger

from invest_advisor_bot.runtime_diagnostics import render_prometheus_metrics

_HEALTH_SERVER: HTTPServer | None = None
_HEALTH_LOCK = threading.Lock()
_HEALTH_DETAILS_PROVIDER: Callable[[], Mapping[str, Any]] | None = None


class HealthCheckHandler(BaseHTTPRequestHandler):
    """Serve a tiny health endpoint for container and platform probes."""

    def do_GET(self) -> None:  # noqa: N802 - stdlib signature
        self._send_health_response(include_body=True)

    def do_HEAD(self) -> None:  # noqa: N802 - stdlib signature
        self._send_health_response(include_body=False)

    def _send_health_response(self, *, include_body: bool) -> None:
        payload = {"status": "ok", "service": "invest-advisor-bot"}
        if self.path not in {"/", "/health", "/diagnostics", "/metrics"}:
            self.send_response(404)
            self.end_headers()
            return
        diagnostics_payload: Mapping[str, Any] | None = None
        details_provider = _HEALTH_DETAILS_PROVIDER
        if callable(details_provider):
            try:
                details = details_provider()
            except Exception as exc:
                details = {"health_details_error": str(exc)}
            if isinstance(details, Mapping):
                details_payload = dict(details)
                diagnostics_payload = details_payload.get("diagnostics") if isinstance(details_payload.get("diagnostics"), Mapping) else None
                if self.path == "/metrics" and isinstance(diagnostics_payload, Mapping):
                    encoded = render_prometheus_metrics(
                        diagnostics_payload.get("runtime") if isinstance(diagnostics_payload.get("runtime"), Mapping) else {}
                    ).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
                    self.send_header("Content-Length", str(len(encoded)))
                    self.end_headers()
                    if include_body:
                        self.wfile.write(encoded)
                    return
                if self.path == "/diagnostics" and isinstance(diagnostics_payload, Mapping):
                    payload = dict(diagnostics_payload)
                else:
                    details_payload.pop("diagnostics", None)
                    payload.update(details_payload)
        if self.path == "/metrics":
            encoded = render_prometheus_metrics(
                diagnostics_payload.get("runtime") if isinstance(diagnostics_payload, Mapping) and isinstance(diagnostics_payload.get("runtime"), Mapping) else {}
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            if include_body:
                self.wfile.write(encoded)
            return

        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        if include_body:
            self.wfile.write(encoded)

    def log_message(self, format: str, *args: object) -> None:
        return


def start_health_check_server(*, host: str, port: int) -> HTTPServer | None:
    """Start a background health-check server exactly once."""

    global _HEALTH_SERVER
    with _HEALTH_LOCK:
        if _HEALTH_SERVER is not None:
            return _HEALTH_SERVER
        try:
            server = HTTPServer((host, port), HealthCheckHandler)
        except OSError as exc:
            logger.warning("Health check server could not bind to {}:{}: {}", host, port, exc)
            return None

        thread = threading.Thread(target=server.serve_forever, daemon=True, name="health-check-server")
        thread.start()
        _HEALTH_SERVER = server
        logger.info("Health check server started on {}:{}", host, port)
        return server


def set_health_details_provider(provider: Callable[[], Mapping[str, Any]] | None) -> None:
    global _HEALTH_DETAILS_PROVIDER
    with _HEALTH_LOCK:
        _HEALTH_DETAILS_PROVIDER = provider


def stop_health_check_server() -> None:
    global _HEALTH_SERVER, _HEALTH_DETAILS_PROVIDER
    with _HEALTH_LOCK:
        if _HEALTH_SERVER is None:
            return
        try:
            _HEALTH_SERVER.shutdown()
            _HEALTH_SERVER.server_close()
        finally:
            _HEALTH_SERVER = None
            _HEALTH_DETAILS_PROVIDER = None
