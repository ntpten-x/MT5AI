from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from loguru import logger

_HEALTH_SERVER: HTTPServer | None = None
_HEALTH_LOCK = threading.Lock()


class HealthCheckHandler(BaseHTTPRequestHandler):
    """Serve a tiny health endpoint for container and platform probes."""

    def do_GET(self) -> None:  # noqa: N802 - stdlib signature
        payload = {"status": "ok", "service": "invest-advisor-bot"}
        if self.path not in {"/", "/health"}:
            self.send_response(404)
            self.end_headers()
            return

        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
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


def stop_health_check_server() -> None:
    global _HEALTH_SERVER
    with _HEALTH_LOCK:
        if _HEALTH_SERVER is None:
            return
        try:
            _HEALTH_SERVER.shutdown()
            _HEALTH_SERVER.server_close()
        finally:
            _HEALTH_SERVER = None
