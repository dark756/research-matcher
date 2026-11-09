"""HTTP client: SQLite response cache (2xx only), key redaction, OpenAlex credit guard, S2 limiter + backoff."""
from __future__ import annotations

import hashlib
import json
import random
import re
import sqlite3
import time
import zlib
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import httpx

from .config import load_secrets, load_settings
from .db import connect
from .rate_limit import S2RateLimiter

SECRET_PARAMS = {"api_key", "apikey", "key"}


class CreditGuardError(RuntimeError):
    pass


class HttpError(RuntimeError):
    def __init__(self, status: int, url: str):
        super().__init__(f"HTTP {status} for {url}")
        self.status = status


def redact_url(url: str) -> str:
    """Strip secret query params (api_key=...) so they never reach the cache key or logs."""
    parts = urlsplit(url)
    q = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True) if k.lower() not in SECRET_PARAMS]
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(q), ""))


def redact_text(text: str) -> str:
    return re.sub(r"(api_key=)[^&\s]+", r"\1<redacted>", text)


def cache_key(method: str, url: str, body: Any = None) -> str:
    payload = json.dumps(body, sort_keys=True) if body is not None else ""
    return hashlib.sha256(f"{method.upper()}\n{redact_url(url)}\n{payload}".encode()).hexdigest()


class HttpClient:
    def __init__(self, conn: sqlite3.Connection | None = None):
        self.conn = conn or connect()
        self.settings = load_settings()
        self.secrets = load_secrets()
        self.client = httpx.Client(timeout=60, headers={"User-Agent": "research-matcher/0.1"})
        self._last_openalex = 0.0
        self.s2_limiter = S2RateLimiter(self.conn, self.settings.semantic_scholar["min_interval_s"])

    # --- cache -----------------------------------------------------------
    def _cache_get(self, key: str) -> Any | None:
        row = self.conn.execute("SELECT body FROM http_cache WHERE key=?", (key,)).fetchone()
        return json.loads(zlib.decompress(row[0])) if row else None

    def _cache_put(self, key: str, url: str, status: int, data: Any) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO http_cache VALUES (?,?,?,?,?)",
            (key, redact_url(url), status, zlib.compress(json.dumps(data).encode()), time.time()),
        )

    # --- OpenAlex ----------------------------------------------------------
    def openalex(self, path: str, params: dict[str, Any] | None = None, use_cache: bool = True) -> Any:
        base = self.settings.http["openalex_base_url"]
        params = dict(params or {})
        url = f"{base}{path}?{urlencode(params)}" if params else f"{base}{path}"
        key = cache_key("GET", url)
        if use_cache and (hit := self._cache_get(key)) is not None:
            return hit
        self._guard_credits()
        full = dict(params, mailto=self.secrets.openalex_email)
        if self.secrets.openalex_api_key:
            full["api_key"] = self.secrets.openalex_api_key
        self._throttle_openalex()
        for attempt in range(5):
            r = self.client.get(f"{base}{path}", params=full)
            self._record_credits(r)
            if r.status_code == 429:
                time.sleep(float(r.headers.get("Retry-After", 2 ** (attempt + 1))))
                continue
            if r.status_code >= 500:
                time.sleep(2 ** (attempt + 1))
                continue
            break
        if not r.is_success:
            raise HttpError(r.status_code, redact_url(str(r.url)))
        data = r.json()
        self._cache_put(key, url, r.status_code, data)
        return data

    def _throttle_openalex(self) -> None:
        gap = 1.0 / float(self.settings.http["openalex_max_rps"])
        wait = self._last_openalex + gap - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        self._last_openalex = time.monotonic()

    def _record_credits(self, r: httpx.Response) -> None:
        rem = r.headers.get("X-RateLimit-Remaining")
        if rem is not None:
            self.conn.execute(
                "INSERT INTO rate_limits(service, credits_remaining) VALUES('openalex',?) "
                "ON CONFLICT(service) DO UPDATE SET credits_remaining=excluded.credits_remaining",
                (int(float(rem)),),
            )

    def credits_remaining(self) -> int | None:
        row = self.conn.execute("SELECT credits_remaining FROM rate_limits WHERE service='openalex'").fetchone()
        return row[0] if row else None

    def _guard_credits(self) -> None:
        rem = self.credits_remaining()
        floor = self.settings.harvest["min_credits_remaining"]
        if rem is not None and rem < floor:
            raise CreditGuardError(f"OpenAlex credits remaining {rem} < {floor}; stopping.")

    # --- Semantic Scholar ----------------------------------------------------
    def s2_post(self, path: str, params: dict[str, Any], body: Any) -> Any:
        """POST to S2. Every attempt (including retries) takes a limiter token. Key is header-only."""
        cfg = self.settings.semantic_scholar
        url = f"{cfg['base_url']}{path}?{urlencode(params)}"
        key = cache_key("POST", url, body)
        if (hit := self._cache_get(key)) is not None:
            return hit
        bo = cfg["backoff"]
        headers = {"x-api-key": self.secrets.s2_api_key} if self.secrets.s2_api_key else {}
        for attempt in range(bo["max_attempts"]):
            self.s2_limiter.acquire()
            r = self.client.post(url, json=body, headers=headers)
            if r.is_success:
                data = r.json()
                self._cache_put(key, url, r.status_code, data)
                return data
            if r.status_code not in (429, 500, 502, 503, 504):
                raise HttpError(r.status_code, url)
            delay = r.headers.get("Retry-After")
            wait = float(delay) if delay else min(bo["cap_s"], bo["base_s"] * 2**attempt)
            if bo["jitter"]:
                wait *= random.uniform(0.8, 1.2)
            time.sleep(wait)
        raise HttpError(429, url)

    def get_json(self, url: str, headers: dict[str, str] | None = None) -> Any:
        """Plain cached GET for other services (HF, etc.)."""
        key = cache_key("GET", url)
        if (hit := self._cache_get(key)) is not None:
            return hit
        r = self.client.get(url, headers=headers)
        if not r.is_success:
            raise HttpError(r.status_code, redact_url(url))
        data = r.json()
        self._cache_put(key, url, r.status_code, data)
        return data

    def close(self) -> None:
        self.client.close()
