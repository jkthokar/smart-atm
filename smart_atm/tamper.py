"""Tamper events from external sensors over HTTP.

Any device on the same network can report tampering by sending
`POST /tamper` to this computer, for example:

* a smartphone automation app (such as Tasker or iOS Shortcuts) that sends a
  request when the phone's accelerometer detects a shock, as in the v1 prototype;
* an ESP32 with an SW-420 vibration sensor (see `hardware/esp32_vibration_sensor.ino`);
* a manual test: `curl -X POST http://<pc-ip>:8765/tamper`.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from queue import Empty, Queue
from typing import List, Optional


class TamperListener:
    def __init__(self, host: str = "0.0.0.0", port: int = 8765):
        self.events: "Queue[dict]" = Queue()
        events = self.events

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802 (http.server naming)
                if self.path.rstrip("/") != "/tamper":
                    self.send_error(404)
                    return
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length) if length else b""
                try:
                    payload = json.loads(raw) if raw else {}
                except json.JSONDecodeError:
                    payload = {"raw": raw.decode(errors="replace")}
                events.put({"source": self.client_address[0], **payload})
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"status":"alarm"}')

            def log_message(self, *args):  # keep the console quiet
                pass

        self.server = ThreadingHTTPServer((host, port), Handler)
        self._thread: Optional[threading.Thread] = None

    @property
    def port(self) -> int:
        return self.server.server_address[1]

    def start(self) -> "TamperListener":
        self._thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def poll(self) -> List[dict]:
        """Return all events received since the last call (non-blocking)."""
        out = []
        while True:
            try:
                out.append(self.events.get_nowait())
            except Empty:
                return out

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()
