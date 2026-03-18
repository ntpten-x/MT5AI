import os
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from loguru import logger


class HealthCheckHandler(BaseHTTPRequestHandler):
    """Simple HTTP requests handler to serve health checks."""

    def do_GET(self):
        if self.path in ("/", "/health"):
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status": "healthy", "service": "invest-advisor-bot"}')
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        # Suppress noisy standard HTTP logs to keep logs clean
        pass


def start_health_check_server() -> None:
    """Start light server on background thread responding to Render diagnostics."""
    # Render automatically populates PORT env variable for web services
    port = int(os.environ.get("PORT", 10000))
    
    try:
        server = HTTPServer(("0.0.0.0", port), HealthCheckHandler)
        # Run as a daemon thread so it shuts down with the main app
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        logger.info(f"✅ Health check server started on port {port} (/health)")
    except Exception as e:
        logger.warning(f"⚠️ Could not start health check server (might be running locally): {e}")
