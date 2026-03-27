from __future__ import annotations

import asyncio
import json
import threading
from concurrent.futures import Future
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Callable, Mapping

from loguru import logger
from telegram import Update

from invest_advisor_bot.runtime_diagnostics import render_prometheus_metrics

_HEALTH_SERVER: HTTPServer | None = None
_HEALTH_LOCK = threading.Lock()
_HEALTH_DETAILS_PROVIDER: Callable[[], Mapping[str, Any]] | None = None
_WEBHOOK_DISPATCHER: Callable[[dict[str, Any]], Future[Any] | None] | None = None
_WEBHOOK_PATH: str | None = None
_WEBHOOK_SECRET_TOKEN: str | None = None


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

    def do_POST(self) -> None:  # noqa: N802 - stdlib signature
        webhook_path = _WEBHOOK_PATH or ""
        dispatcher = _WEBHOOK_DISPATCHER
        secret_token = (_WEBHOOK_SECRET_TOKEN or "").strip()
        if not webhook_path or self.path != webhook_path or not callable(dispatcher):
            self.send_response(404)
            self.end_headers()
            return
        if secret_token:
            received_secret = str(self.headers.get("X-Telegram-Bot-Api-Secret-Token") or "").strip()
            if received_secret != secret_token:
                self.send_response(403)
                self.end_headers()
                return
        content_length = int(self.headers.get("Content-Length") or 0)
        if content_length <= 0:
            self.send_response(400)
            self.end_headers()
            return
        try:
            payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            self.send_response(400)
            self.end_headers()
            return
        try:
            future = dispatcher(payload)
            if future is not None:
                def _log_future_error(done_future: Future[Any]) -> None:
                    try:
                        done_future.result()
                    except Exception as exc:  # pragma: no cover - defensive async logging
                        logger.warning("Webhook update enqueue failed asynchronously: {}", exc)

                future.add_done_callback(_log_future_error)
        except Exception as exc:
            logger.warning("Webhook dispatch failed: {}", exc)
            self.send_response(500)
            self.end_headers()
            return
        self.send_response(200)
        self.end_headers()

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


def set_telegram_webhook_dispatcher(
    *,
    application: object | None,
    loop: asyncio.AbstractEventLoop | None,
    path: str | None,
    secret_token: str | None = None,
) -> None:
    global _WEBHOOK_DISPATCHER, _WEBHOOK_PATH, _WEBHOOK_SECRET_TOKEN
    normalized_path = "/" + str(path or "").strip().strip("/") if str(path or "").strip() else None
    with _HEALTH_LOCK:
        _WEBHOOK_PATH = normalized_path
        _WEBHOOK_SECRET_TOKEN = str(secret_token or "").strip() or None
        if application is None or loop is None or normalized_path is None:
            _WEBHOOK_DISPATCHER = None
            return

        def _dispatch(payload: dict[str, Any]) -> Future[Any] | None:
            bot = getattr(application, "bot", None)
            update_queue = getattr(application, "update_queue", None)
            if bot is None or update_queue is None:
                return None
            update = Update.de_json(payload, bot)
            return asyncio.run_coroutine_threadsafe(update_queue.put(update), loop)

        _WEBHOOK_DISPATCHER = _dispatch


def stop_health_check_server() -> None:
    global _HEALTH_SERVER, _HEALTH_DETAILS_PROVIDER, _WEBHOOK_DISPATCHER, _WEBHOOK_PATH, _WEBHOOK_SECRET_TOKEN
    with _HEALTH_LOCK:
        try:
            if _HEALTH_SERVER is not None:
                _HEALTH_SERVER.shutdown()
                _HEALTH_SERVER.server_close()
        finally:
            _HEALTH_SERVER = None
            _HEALTH_DETAILS_PROVIDER = None
            _WEBHOOK_DISPATCHER = None
            _WEBHOOK_PATH = None
            _WEBHOOK_SECRET_TOKEN = None
