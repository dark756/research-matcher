"""Global S2 rate limiter: >= min_interval between ANY two S2 requests, across threads and processes.

State lives in the `rate_limits` table (last_request_at). The slot is *reserved* under a
BEGIN IMMEDIATE lock (we write our planned send time), then we sleep outside the lock, so
concurrent processes queue up one after another and never fire within min_interval.
"""
from __future__ import annotations

import sqlite3
import threading
import time
from typing import Callable

from .db import connect


class S2RateLimiter:
    def __init__(
        self,
        conn: sqlite3.Connection | None = None,
        min_interval_s: float = 1.05,
        service: str = "s2",
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.conn = conn or connect()
        self.min_interval = min_interval_s
        self.service = service
        self._clock, self._sleep = clock, sleep
        self._lock = threading.Lock()  # one S2 call at a time inside a process

    def acquire(self) -> float:
        """Block until a request may be sent. Returns the reserved send time."""
        with self._lock:
            c = self.conn
            c.execute("BEGIN IMMEDIATE")
            try:
                row = c.execute(
                    "SELECT last_request_at FROM rate_limits WHERE service=?", (self.service,)
                ).fetchone()
                last = row[0] if row and row[0] else 0.0
                send_at = max(self._clock(), last + self.min_interval)
                c.execute(
                    "INSERT INTO rate_limits(service, last_request_at) VALUES(?,?) "
                    "ON CONFLICT(service) DO UPDATE SET last_request_at=excluded.last_request_at",
                    (self.service, send_at),
                )
                c.execute("COMMIT")
            except BaseException:
                c.execute("ROLLBACK")
                raise
            wait = send_at - self._clock()
            if wait > 0:
                self._sleep(wait)
            return send_at
